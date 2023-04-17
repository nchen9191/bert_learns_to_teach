import torch
import json

with open("./glue_meta_data.json") as j:
    GLUE_META_DATA = json.load(j)
