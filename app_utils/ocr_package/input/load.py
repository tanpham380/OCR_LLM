import os
from PIL import Image
import json

def get_name_from_path(path):
    return os.path.basename(path).split(".")[0]



def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    name = get_name_from_path(image_path)
    return [image], [name]


def load_lang_file(lang_path, names):
    with open(lang_path, "r") as f:
        lang_dict = json.load(f)
    return [lang_dict[name].copy() for name in names]
