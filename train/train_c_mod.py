import torch
import torchvision
from torch import nn, optim
from transformers import AutoTokenizer, Owlv2TextModel, Owlv2VisionModel, modeling_outputs
from warmup_scheduler import GradualWarmupScheduler

import sys
import os
from dataclasses import dataclass

from gdf import GDF, EpsilonTarget, CosineSchedule
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, P2LossWeight, AdaptiveLossWeight
from danbooru import db
from torchtools.transforms import SmartCrop

from modules.effnet import EfficientNetEncoder
from modules.stage_c import StageC
from modules.stage_c_mod_bitnet import StageCTransformer, TimestepEmbedder, LatentDecoder, LatentEncoder, BitFeedForward
from modules.bitnet.attn import MultiheadAttention, MultiModalCrossAttention
from modules.stage_c import ResBlock, AttnBlock, TimestepBlock, FeedForwardBlock
from modules.previewer import Previewer

from train.base import DataCore, TrainingCore

from core import WarpCore
from core.utils import EXPECTED, EXPECTED_TRAIN, load_or_fail

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
import torch.multiprocessing as mp
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from contextlib import contextmanager
import numpy as np


class WurstCore(TrainingCore, DataCore, WarpCore):
    @dataclass(frozen=True)
    class Config(TrainingCore.Config, DataCore.Config, WarpCore.Config):
        # TRAINING PARAMS
        lr: float = EXPECTED_TRAIN
        warmup_updates: int = EXPECTED_TRAIN
        dtype: str = None

        # MODEL VERSION
        model_version: str = EXPECTED  # 3.6B or 1B
        clip_image_model_name: str = 'google/owlv2-base-patch16-ensemble'
        clip_text_model_name: str = 'google/owlv2-base-patch16-ensemble'

        # CHECKPOINT PATHS
        effnet_checkpoint_path: str = EXPECTED
        previewer_checkpoint_path: str = EXPECTED
        generator_checkpoint_path: str = None

        # gdf customization
        adaptive_loss_weight: str = None

    @dataclass(frozen=True)
    class Models(TrainingCore.Models, DataCore.Models, WarpCore.Models):
        effnet: nn.Module = EXPECTED
        previewer: nn.Module = EXPECTED

    @dataclass(frozen=True)
    class Schedulers(WarpCore.Schedulers):
        generator: any = None

    @dataclass(frozen=True)
    class Extras(TrainingCore.Extras, DataCore.Extras, WarpCore.Extras):
        gdf: GDF = EXPECTED
        sampling_configs: dict = EXPECTED
        effnet_preprocess: torchvision.transforms.Compose = EXPECTED

    info: TrainingCore.Info
    config: Config

    def setup_extras_pre(self) -> Extras:
        gdf = GDF(
            schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
            input_scaler=VPScaler(), target=EpsilonTarget(),
            noise_cond=CosineTNoiseCond(),
            loss_weight=AdaptiveLossWeight() if self.config.adaptive_loss_weight is True else P2LossWeight(),
        )
        sampling_configs = {"cfg": 5, "sampler": DDPMSampler(gdf), "shift": 1, "timesteps": 20}

        if self.info.adaptive_loss is not None:
            gdf.loss_weight.bucket_ranges = torch.tensor(self.info.adaptive_loss['bucket_ranges'])
            gdf.loss_weight.bucket_losses = torch.tensor(self.info.adaptive_loss['bucket_losses'])

        effnet_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        ])

        clip_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(960, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(960),
            torchvision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        if self.config.training:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.config.image_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
                SmartCrop(self.config.image_size, randomize_p=0.3, randomize_q=0.2)
            ])
        else:
            transforms = None

        return self.Extras(
            gdf=gdf,
            sampling_configs=sampling_configs,
            transforms=transforms,
            effnet_preprocess=effnet_preprocess,
            clip_preprocess=clip_preprocess
        )
    
    def webdataset_path(self):
        webdataset_paths = [self.config.webdataset_path.format(str(i).rjust(4, "0")) for i in range(1128)]
        if not self.config.use_fsdp:
            base_size = 1127//self.n_gpu_per_node
            webdataset_paths = webdataset_paths[self.rank*base_size:(self.rank+1)*base_size]
        return webdataset_paths
    
    def webdataset_preprocessors(self, extras: Extras):
        return [
            ('jpg;png;webp', torchvision.transforms.ToTensor() if self.config.multi_aspect_ratio is not None else extras.transforms, 'images'),
            ("__key__", db.get_tags, "captions"),
            ("__key__", db.get_embeddings, "embeddings")
        ]
    
    

    def get_conditions(self, batch: dict, models: Models, extras: Extras, is_eval=False, is_unconditional=False,
                       eval_image_embeds=False, return_fields=None):
        if is_unconditional:
            embeddings = db.owl_embeds["uncond"].expand(len(batch["embeddings"]), 1, 512)
        else:
            embeddings = []
            for x in batch["embeddings"]:
                try:
                    embeddings.append(torch.cat(x, dim=0))
                except:
                    embeddings.append(db.owl_embeds["uncond"].unsqueeze(0))
            embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value=0.)
        return {"text_emb": embeddings.to(self.device)}

    def setup_models(self, extras: Extras) -> Models:
        dtype = getattr(torch, self.config.dtype) if self.config.dtype else torch.float32

        # EfficientNet encoder
        effnet = EfficientNetEncoder()
        effnet_checkpoint = load_or_fail(self.config.effnet_checkpoint_path)
        effnet.load_state_dict(effnet_checkpoint if 'state_dict' not in effnet_checkpoint else effnet_checkpoint['state_dict'])
        effnet.eval().requires_grad_(False).to(self.device)
        del effnet_checkpoint

        # Previewer
        previewer = Previewer()
        previewer_checkpoint = load_or_fail(self.config.previewer_checkpoint_path)
        previewer.load_state_dict(previewer_checkpoint if 'state_dict' not in previewer_checkpoint else previewer_checkpoint['state_dict'])
        previewer.eval().requires_grad_(False).to(self.device)
        del previewer_checkpoint

        @contextmanager
        def dummy_context():
            yield None

        loading_context = dummy_context if self.config.training else init_empty_weights

        # Diffusion models
        with loading_context():
            generator_ema = None
            if self.config.model_version == '3.6B':
                generator = StageC()
                if self.config.ema_start_iters is not None:
                    generator_ema = StageC()
            elif self.config.model_version == '1B':
                generator = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]])
                if self.config.ema_start_iters is not None:
                    generator_ema = StageC(c_cond=1536, c_hidden=[1536, 1536], nhead=[24, 24], blocks=[[4, 12], [12, 4]])
            elif self.config.model_version == "NTT":
                generator = StageCTransformer()
                if self.config.ema_start_iters is not None:
                    generator_ema = StageCTransformer()
            else:
                raise ValueError(f"Unknown model version {self.config.model_version}")

        if self.config.generator_checkpoint_path is not None:
            if loading_context is dummy_context:
                generator.load_state_dict(load_or_fail(self.config.generator_checkpoint_path))
            else:
                for param_name, param in load_or_fail(self.config.generator_checkpoint_path).items():
                    set_module_tensor_to_device(generator, param_name, "cpu", value=param)
        generator = generator.to(dtype).to(self.device)
        generator = self.load_model(generator, 'generator')

        if generator_ema is not None:
            if loading_context is dummy_context:
                generator_ema.load_state_dict(generator.state_dict())
            else:
                for param_name, param in generator.state_dict().items():
                    set_module_tensor_to_device(generator_ema, param_name, "cpu", value=param)
            generator_ema = self.load_model(generator_ema, 'generator_ema')
            generator_ema.to(dtype).to(self.device).eval().requires_grad_(False)

        if self.config.use_fsdp:
            fsdp_auto_wrap_policy = ModuleWrapPolicy([TimestepEmbedder, LatentDecoder, LatentEncoder, BitFeedForward, MultiheadAttention, MultiModalCrossAttention])
            generator = FSDP(generator, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)
            if generator_ema is not None:
                generator_ema = FSDP(generator_ema, **self.fsdp_defaults, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=self.device)

        # CLIP encoders
        tokenizer = AutoTokenizer.from_pretrained(self.config.clip_text_model_name)
        text_model = Owlv2TextModel.from_pretrained(self.config.clip_text_model_name).requires_grad_(False).to(dtype).to(self.device)
        image_model = Owlv2VisionModel.from_pretrained(self.config.clip_image_model_name).requires_grad_(False).to(dtype).to(self.device)

        return self.Models(
            effnet=effnet, previewer=previewer,
            generator=generator, generator_ema=generator_ema,
            tokenizer=tokenizer, text_model=text_model, image_model=image_model
        )

    def setup_optimizers(self, extras: Extras, models: Models) -> TrainingCore.Optimizers:
        optimizer = optim.AdamW(models.generator.parameters(), lr=self.config.lr)  # , eps=1e-7, betas=(0.9, 0.95))
        optimizer = self.load_optimizer(optimizer, 'generator_optim',
                                        fsdp_model=models.generator if self.config.use_fsdp else None)
        return self.Optimizers(generator=optimizer)

    def setup_schedulers(self, extras: Extras, models: Models, optimizers: TrainingCore.Optimizers) -> Schedulers:
        scheduler = GradualWarmupScheduler(optimizers.generator, multiplier=1, total_epoch=self.config.warmup_updates)
        scheduler.last_epoch = self.info.total_steps
        return self.Schedulers(generator=scheduler)

    # Training loop --------------------------------
    def forward_pass(self, data: WarpCore.Data, extras: Extras, models: Models):
        batch = next(data.iterator)
        B, C, W, H = batch["images"].shape
        self.last_shape = (W, H)

        with torch.no_grad():
            conditions = self.get_conditions(batch, models, extras)
            latents = self.encode_latents(batch, models, extras)
            noised, noise, target, logSNR, noise_cond, loss_weight = extras.gdf.diffuse(latents, shift=1, loss_shift=1)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = models.generator(noised, noise_cond, **conditions)
            loss = nn.functional.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
            loss_adjusted = (loss * loss_weight).mean() / self.config.grad_accum_steps

        if isinstance(extras.gdf.loss_weight, AdaptiveLossWeight):
            extras.gdf.loss_weight.update_buckets(logSNR, loss)

        return loss, loss_adjusted

    def backward_pass(self, update, loss, loss_adjusted, models: Models, optimizers: TrainingCore.Optimizers, schedulers: Schedulers):
        if update:
            loss_adjusted.backward()
            grad_norm = nn.utils.clip_grad_norm_(models.generator.parameters(), 1.0)
            optimizers_dict = optimizers.to_dict()
            for k in optimizers_dict:
                if k != 'training':
                    optimizers_dict[k].step()
            schedulers_dict = schedulers.to_dict()
            for k in schedulers_dict:
                if k != 'training':
                    schedulers_dict[k].step()
            for k in optimizers_dict:
                if k != 'training':
                    optimizers_dict[k].zero_grad(set_to_none=True)
            self.info.total_steps += 1
        else:
            loss_adjusted.backward()
            grad_norm = torch.tensor(0.0).to(self.device)

        return grad_norm

    def models_to_save(self):
        return ['generator', 'generator_ema']

    def encode_latents(self, batch: dict, models: Models, extras: Extras) -> torch.Tensor:
        images = batch['images'].to(self.device)
        return models.effnet(extras.effnet_preprocess(images))

    def decode_latents(self, latents: torch.Tensor, batch: dict, models: Models, extras: Extras) -> torch.Tensor:
        return models.previewer(latents)


def run(rank, config_file_path, n_gpu_per_node=1, dataset=None):
    warpcore = WurstCore(
        config_file_path=config_file_path,
        device="cuda"
    )
    if n_gpu_per_node == 1:
        single_gpu = True
    else:
        single_gpu = False
    warpcore.__call__(rank, single_gpu, n_gpu_per_node=n_gpu_per_node, dataset=dataset)

def main():
    print("Launching Script")
    db.setup()
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    config_file_path = sys.argv[1] if len(sys.argv) > 1 else None
    gpus = torch.cuda.device_count()

    # RUN TRAINING
    if gpus != 1:
        mp.spawn(
            run,
            (
                config_file_path, 
                gpus
            ),
            nprocs=gpus,
        )
    else:
        run(0, config_file_path)
    

if __name__ == "__main__":
    main()