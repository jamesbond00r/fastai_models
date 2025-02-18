# Standard library imports
import os
import time
import json

# Third-party library imports
import requests
from PIL import Image

# FastAI-related imports
from fastai.vision.all import *
from fastai.data.all import *
from fastai.vision.augment import Resize, ResizeMethod

# DuckDuckGo Search API
from duckduckgo_search import DDGS  # Updated API

# Fastcore and Fastdownload
from fastcore.foundation import L  # Importing L for list handling
from fastdownload import download_url


thumbnail = 256
# Define my function
# Max images is an optional paramter that defaults to 200 images
def search_images(keywords, max_images=200):
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')
# DDGS() creates instance of duckduckgo search class
# L() converts to a fastcore list
# .iteamgot(image) fastcore method extracts the image filed 

def ResizeToTHumb(file, size):
    im = Image.open(file)
    im_resized = im.resize((size, size), Image.LANCZOS)
    im_resized.save(file)


# Calls the duck duck go seach 
urls = search_images('Corgi photos', max_images=10)
# Print out a sample
print(urls[1])

# name the image and download it
dest = 'corgi.jpg'
download_url(urls[1], dest, show_progress=False)

# open the image and turn it into a thumb nail
ResizeToTHumb(dest, thumbnail)



# # Download things that are not a corgi picture (foxes)
# # Calls the duck duck go seach for foxes
# urlsFox = search_images('fox photos', max_images=10)
# # Print out a sample
# print(urlsFox[1])
# destFox = 'fox.jpg'
# download_url(urlsFox[1], destFox, show_progress=False)

# ResizeToTHumb(destFox, thumbnail)


searches = 'corgi','fox'
path = Path('corgi_or_not')
for index, o in enumerate(searches):
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    urls=search_images(f'{o} photo', 30)
    for url in urls:
        try:
            download_url( url, dest, show_progress=False)
            print(url)
            time.sleep(5)
            base = os.path.basename(url)
            file = str(dest) + "/" + str(base)
            ResizeToTHumb(file, thumbnail)
        except Exception as e:
            print(f"Failed to download {url}. Error: {e}")
            continue  # Skip to the next URL if download fails


failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

# print(path.ls()) 
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.1, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, ResizeMethod.Squish)]  # Corrected Syntax
).dataloaders(path, bs=32)

for batch in dls.train:
    print("Train Batch:", batch)
    break  # Stop after one batch

for batch in dls.valid:
    print("Valid Batch:", batch)
    break  # Stop after one batch

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

is_corgi,_,probs = learn.predict(PILImage.create('corgi.jpg'))
print(f"This is a: {is_corgi}.")
print(f"Probability it's a corgi: {probs[0]:.4f}")