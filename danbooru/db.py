import sqlite3
import os
import requests
import tqdm
from peewee import *
import torch
import typing


def download(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm.tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


class MemoryConnection(sqlite3.Connection):
    def __init__(self, dbname, *args, **kwargs):
        load_conn = sqlite3.connect(dbname)
        super(MemoryConnection, self).__init__(":memory:", *args, **kwargs)
        load_conn.backup(self)
        load_conn.close()


class SqliteMemDatabase(SqliteDatabase):
    def __init__(self, database, *args, **kwargs) -> None:
        self.dbname = database
        super().__init__(database, *args, factory=MemoryConnection, **kwargs)

    def reload(self, dbname=None):
        if dbname is None:
            dbname = self.dbname
        load_conn = sqlite3.connect(dbname)
        try:
            load_conn.backup(self._state.conn)
        finally:
            load_conn.close()

    def save(self, dbname=None):
        if dbname is None:
            dbname = self.dbname
        save_conn = sqlite3.connect(dbname)
        try:
            self._state.conn.backup(save_conn)
        finally:
            save_conn.close()


db: SqliteDatabase = None
tag_cache_map = {}


def get_tag_by_id(id):
    if id not in tag_cache_map:
        tag_cache_map[id] = Tag.get_by_id(id)
    return tag_cache_map[id]


class EnumField(IntegerField):
    def __init__(self, enum_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enum_list = enum_list
        self.enum_map = {value: index for index, value in enumerate(enum_list)}

    def db_value(self, value):
        if isinstance(value, str):
            return self.enum_map[value]
        assert isinstance(value, int)
        return value

    def python_value(self, value):
        if value is not None:
            return self.enum_list[value]


class BaseModel(Model):
    class Meta:
        database = db


class Tag(BaseModel):
    id = IntegerField(primary_key=True)
    name = CharField(unique=True)
    type = EnumField(["general", "artist", "character", "copyright", "meta"])
    popularity = IntegerField()
    _posts: ManyToManyField
    _posts_cache = None

    @property
    def posts(self):
        if self._posts_cache is None:
            self._posts_cache = list(self._posts)
        return self._posts_cache

    def __str__(self):
        return f"<Tag '{self.name}'>"

    def __repr__(self):
        from objprint import objstr

        return f"<Tag|#{self.id}|{self.name}|{self.type[:2]}>"


class Post(BaseModel):
    id = IntegerField(primary_key=True)
    created_at = CharField()
    uploader_id = IntegerField()
    source = CharField()
    md5 = CharField(null=True)
    parent_id = IntegerField(null=True)
    has_children = BooleanField()
    is_deleted = BooleanField()
    is_banned = BooleanField()
    pixiv_id = IntegerField(null=True)
    has_active_children = BooleanField()
    bit_flags = IntegerField()
    has_large = BooleanField()
    has_visible_children = BooleanField()

    image_width = IntegerField()
    image_height = IntegerField()
    file_size = IntegerField()
    file_ext = CharField()

    rating = EnumField(["general", "sensitive", "questionable", "explicit"])
    score = IntegerField()
    up_score = IntegerField()
    down_score = IntegerField()
    fav_count = IntegerField()

    file_url = CharField()
    large_file_url = CharField()
    preview_file_url = CharField()

    _tags: ManyToManyField
    _tags_cache = None
    _tag_list = TextField(column_name="tag_list")

    tag_count = IntegerField()
    tag_count_general = IntegerField()
    tag_count_artist = IntegerField()
    tag_count_character = IntegerField()
    tag_count_copyright = IntegerField()
    tag_count_meta = IntegerField()

    @property
    def tag_list(self):
        if self._tags_cache is None:
            self._tags_cache = list(self._tags)
        return self._tags_cache

    @property
    def tag_list_general(self):
        return [tag for tag in self.tag_list if tag.type == "general"]

    @property
    def tag_list_artist(self):
        return [tag for tag in self.tag_list if tag.type == "artist"]

    @property
    def tag_list_character(self):
        return [tag for tag in self.tag_list if tag.type == "character"]

    @property
    def tag_list_copyright(self):
        return [tag for tag in self.tag_list if tag.type == "copyright"]

    @property
    def tag_list_meta(self):
        return [tag for tag in self.tag_list if tag.type == "meta"]


class PostTagRelation(BaseModel):
    post = ForeignKeyField(Post, backref="post_tags")
    tag = ForeignKeyField(Tag, backref="tag_posts")


tags = ManyToManyField(Tag, backref="_posts", through_model=PostTagRelation)
tags.bind(Post, "_tags", set_attribute=True)
owl_embeds = None
keys = []

def load_db(db_file: str):
    global db
    db = SqliteDatabase(db_file)
    Post._meta.database = db
    Tag._meta.database = db
    PostTagRelation._meta.database = db
    db.connect()

def setup(path="danbooru2023.db", embedding="embeds.pt"):
    if not os.path.isfile(path):
        download("https://huggingface.co/datasets/KBlueLeaf/danbooru2023-sqlite/resolve/main/danbooru2023.db", path)
    if not os.path.isfile(embedding):
        download("https://huggingface.co/junjuice0/test/resolve/main/embeds.pt", embedding)
    load_db("danbooru2023.db")
    global owl_embeds, keys
    owl_embeds = torch.load(embedding)
    keys = list(owl_embeds.keys())


def get_quality_tag(score: int):
    if score > 150:
        return "masterpiece"
    elif score > 100:
        return "best quality"
    elif score > 75:
        return "high quality"
    elif score > 25:
        return "medium quality"
    elif score > 0:
        return None
    elif score > -5:
        return "low quality"
    else:
        return "worst quality"

def get_tags(id, embedding: bool=False, formatting: bool=True, quality: bool=True, ignore: list[str] = []):
    id = int(id)
    try:
        post: Post = Post.get_by_id(id)
    except:
        if formatting:
            return ""
        else:
            [""]
    
    quality_tag = get_quality_tag(post.score)
    
    if embedding:
        tags_raw = [x.id for x in post.tag_list_general + post.tag_list_character]
        global owl_embeds, keys
        if quality_tag and quality:
            tags = [owl_embeds[quality_tag].unsqueeze(dim=0), ]
        else:
            tags = []
        for tag in tags_raw:
            if str(tag) in keys:
                print(str(tag))
                tags.append(owl_embeds[str(tag)].unsqueeze(dim=0))
        return torch.cat(tags, dim=0)
    else:
        tags_raw = [x.name for x in post.tag_list_general + post.tag_list_character]
        if quality_tag and quality:
            tags = [quality_tag, ]
        else:
            tags = []
        for tag in tags_raw:
            if not tag in ignore:
                tags.append(tag)
        if formatting:
            tag_str = ""
            for tag in tags:
                tag_str += tag + ", "
            tag_str = tag_str.removesuffix(", ")
            return tag_str
        else:
            return tags
        
def get_embeddings(id: int):
    return get_tags(id, True)
    
def get_size(id: int):
    post = Post.get_by_id(id)
    return (post.image_width, post.image_height)