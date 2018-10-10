#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title data_scraping.py
@author: Tuan Le
@email: tuanle@hotmail.de
This script uses python3 urllib.request and BeautifulSoup4 library to scrape images from WikiArts
"""

## Import libraries
import urllib.request as url_r
from bs4 import BeautifulSoup
import os
import itertools
import multiprocessing
from multiprocessing.dummy import Pool
import pandas as pd
import re

genre_to_scrape = "yakusha-e"

os.chdir("../data")

### Get all available genres:
all_genre_url = "https://www.wikiart.org/en/paintings-by-genre/"
all_genre_page = url_r.urlopen(all_genre_url)
all_genre_page_parsed = BeautifulSoup(all_genre_page, "lxml")
all_genres = all_genre_page_parsed.findAll(class_="dottedItem")
# Iterate over the genres and save them with they current number of artists into a DataFrame
genres = []
num_artists = []
# Iterate through all_genres and extract the anchor <a> tag storing genre and <sup> tag storing the number of artist for genres:
for el in all_genres:
    #print(el.find("a").contents[0])
    #add some preprocessing for string characters: Remove brackets '()' and add '-' where whitespace is within the text for url
    genres.append(el.find("a").contents[0].replace("(","").replace(")","").rstrip().replace(" ", "-"))
    #print(el.find("sup").contents[0])
    num_artists.append(el.find("sup").contents[0])
    
# Store the genres with number of artists into pandas DataFrame:
all_genres_df = pd.DataFrame({"genres":genres, "number of artists":num_artists})
#print(genres)

# Variable genre_to_scrape can be any from the list mentioned below:
"""
['abstract',
 'advertisement',
 'allegorical-painting',
 'animal-painting',
 'animation',
 'architecture',
 "artist's-book",
 'battle-painting',
 'bijinga',
 'bird-and-flower-painting',
 'calligraphy',
 'capriccio',
 'caricature',
 'cityscape',
 'cloudscape',
 'design',
 'figurative',
 'flower-painting',
 'genre-painting',
 'graffiti',
 'history-painting',
 'illustration',
 'installation',
 'interior',
 'landscape',
 'literary-painting',
 'marina',
 'miniature',
 'mosaic',
 'mythological-painting',
 'nude-painting-nu',
 'panorama',
 'pastorale',
 'performance',
 'photo',
 'portrait',
 'poster',
 'quadratura',
 'religious-painting',
 'sculpture',
 'self-portrait',
 'shan-shui',
 'sketch-and-study',
 'still-life',
 'symbolic-painting',
 'tapestry',
 'tessellation',
 "trompe-l'œil",
 'tronie',
 'urushi-e',
 'vanitas',
 'veduta',
 'video',
 'wildlife-painting',
 'yakusha-e']
"""

def get_artist_url(genre = genre_to_scrape):
    """
    Retrieves the artist relative path url from wikiart.org/en according to the selected genre
    
    Input:
        genre: [str] character string from one of the list above
    Output:
        url_list: [list] containing relative path urls, i.e one list element is 'wassily-kandinsky' for abstract genre
    """
    url_genre = "https://www.wikiart.org/en/paintings-by-genre/" + genre
    soup =  BeautifulSoup(url_r.urlopen(url_genre), "lxml")
    all_artists = soup.findAll("ul", {"class": "artists-group-list"})
    all_artists_txt = str(all_artists)
    all_artists_txt = all_artists_txt.replace('[<ul class="artists-group-list">', '')
    all_artists_txt = all_artists_txt.replace('</ul>]', '').strip()
    all_artists_txt = all_artists_txt.split("\n")
    no_artists = len(all_artists_txt)
    url_list = [None] * no_artists
    for i in range(no_artists):
        url = re.split('\\bartistUrl=\\b', all_artists_txt[i])[1]
        url = url[0:url.find(" ")]
        #url = "https://www.wikiart.org/en/" + url
        url_list[i] = url
    return url_list


def url_helper(html_string):
    """
    Extracts the image url from html file
    """
    try:
        url = re.search("(?P<url>https?://[^\s]+)", html_string).group("url")
        url = url.replace(">", "")
        url = url.replace("]", "")
        url = url.replace('"', "")
        url = url.replace("!PinterestSmall.jpg", "")
        url = url[0:len(url)-1]
        return url
    except:
        print("Could not retrieve url from artist:" + html_string)
        
    

def downloader(artist_url, genre):
    """
    Downloads images from each artist
    Input:
        artist_url: [str] i.e "wassily-kandinsky" will be concatenated as "https://www.wikiart.org/en/wassily-kandinsky"
        genre: passed in script beginning, genre_to_scrape. Needed for folder structure where images ought to be saved
    """
    url = "https://www.wikiart.org/en/" + artist_url
    artist_soup = BeautifulSoup(url_r.urlopen(url), "lxml")
    artist_imgs = artist_soup.findAll("img")
    artist_imgs = str(artist_imgs).split(',')
    #remove first image as its photo of artist
    artist_imgs = artist_imgs[1:len(artist_imgs)]
    artist_imgs = list(map(url_helper, artist_imgs))
    j = 0 
    for file in artist_imgs:
        try:
            path_to_be_saved = genre + "/" + artist_url + "_" + str(j) + ".jpg"
            url_r.urlretrieve(file, path_to_be_saved)
        except:
            print("failed downloading " + str(file))	
        j = j + 1
    

def main(genre): 
    """ 
    Main python function
    Calls predefined functions and distributes the artists-images-to-be-downloaded onto ncores-1 jobs.
    """
    pool_of_threads = Pool(multiprocessing.cpu_count() - 1)
    url_list = get_artist_url(genre_to_scrape)
    pool_of_threads.starmap(downloader, zip(url_list, itertools.repeat(genre)))	
    pool_of_threads.close()

### Start executions ###
if not os.path.exists(os.getcwd() + genre_to_scrape):
    os.mkdir(genre_to_scrape)

print("Building a list of all the paintings to be downloaded from genre:" + genre_to_scrape + "\n This might take a few minutes...")

## Call main python function
main(genre = genre_to_scrape)
print("Downloaded images are at:" + os.getcwd() + "/" + genre_to_scrape)

## END ##