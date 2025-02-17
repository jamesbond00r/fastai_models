import time, json
import os
from duckduckgo_search import DDGS #DuckDuckGo has changed the api so we need to update 
from fastcore.all import *
from fastdownload import download_url
from fastai import *
from PIL import Image



# Define my function
# Max images is an optional paramter that defaults to 200 images
def search_images(keywords, max_images=200):
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')
# DDGS() creates instance of duckduckgo search class
# L() converts to a fastcore list
# .iteamgot(image) fastcore method extracts the image filed 

# Calls the duck duck go seach 
urls = search_images('Corgi photos', max_images=10)
# Print out a sample
print(urls[1])


# name the image and download it
dest = 'corgi.jpg'
download_url(urls[1], dest, show_progress=False)

# open the image and turn it into a thumb nail
im = Image.open(dest)
# Resize the image to exactly 256x256 pixels (this may distort the image if it's not square)
im_resized = im.resize((256, 256), Image.LANCZOS)
# Save the resized image
im_resized.save("corgi.jpg")


# Download things that are not a corgi picture (foxes)
# Calls the duck duck go seach for foxes
urlsFox = search_images('fox photos', max_images=10)
# Print out a sample
print(urlsFox[1])
destFox = 'fox.jpg'
download_url(urlsFox[1], destFox, show_progress=False)
imFox = Image.open(destFox)
im_resizedFox = imFox.resize((256, 256), Image.LANCZOS)
im_resizedFox.save("fox.jpg")
# download_url(search_images('fox photos', max_images=1)[0], 'fox.jpg', show_progress=False)
# Image.open('fox.jpg').resize((256, 256), Image.LANCZOS).save("fox.jpg")

