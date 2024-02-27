from transformers import Owlv2TextModel, Owlv2Processor, AutoTokenizer
import json
import torch
from torch import nn
import tqdm
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_dict = nn.ParameterDict()
bsz = 512

with open("id_to_str.json") as f:
    data = json.load(f)

keys = list(data.keys())
total = math.ceil(len(keys)/bsz)
bar = tqdm.tqdm(range(total))
long = {
    2:[],
    3:[],
    4:[],
    5:[],
}

added = 0
skipped = 0

proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", cache_dir="cache")
tokenizer = AutoTokenizer.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2TextModel.from_pretrained("google/owlv2-base-patch16-ensemble", cache_dir="cache").to(device)

with torch.no_grad():
    for i in bar:
        if i == total -1:
            batch = [data[key].replace("_", " ") for key in keys[-(len(keys)%bsz):]]
            ids = [key for key in keys[-(len(keys)%bsz):]]
        else:
            batch = [data[key].replace("_", " ") for key in keys[i*bsz:(i+1)*bsz]]
            ids = [key for key in keys[i*bsz:(i+1)*bsz]]
        tokenized = tokenizer(batch)
        del_batch = []
        for k in range(len(ids)):
            if len(tokenized[k]) > 16:
                bar.write("Too long tag at ID={} tag:{} seq:{}".format(ids[k], batch[k], math.ceil((len(tokenized[k])-2)/14)))
                long[math.ceil((len(tokenized[k])-2)/14)].append(ids[k])
                del_batch.append(k)
                skipped += 1
            else:
                added += 1
        for j, k in enumerate(del_batch):
            del batch[k-j], ids[k-j]
        batch = proc(text=batch, return_tensors="pt")
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        output = model(**batch)
        for k, key in enumerate(ids):
            embed_dict[str(key)] = output.pooler_output[k, :].to("cpu")
        bar.set_postfix({"added": added, "skipped": skipped})


print(long)
print("ADDED {}".format(added))
print("SKIPPED {}".format(skipped))
print("TOTAL {}".format(len(data)))
torch.save(embed_dict.state_dict(), "embeds.pt")