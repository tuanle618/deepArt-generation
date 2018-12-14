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
import re
import sys
import time
import random
import traceback

####################### Get initial informtation about genre and styles to select #######################

####################### Version 1 #######################
## going via genre ##

### Get all available genres:
all_genre_url = "https://www.wikiart.org/en/paintings-by-genre/"
all_genre_page_soup = BeautifulSoup(url_r.urlopen(all_genre_url), "lxml")
all_genres = all_genre_page_soup.findAll(class_="dottedItem")
# Iterate over the genres and save them with they current number of artists into a DataFrame
genres = []
num_artists = []
# Iterate through all_genres and extract the anchor <a> tag storing genre and <sup> tag storing the number of artist for genres:
for el in all_genres:
    genres.append(el.find("a").contents[0].replace("(","").replace(")","").rstrip().replace(" ", "-"))

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

genre_to_scrape = "yakusha-e"


####################### Version 2 #######################
## going via style        
        
### Add more style genres alltogether via wikiart API https://www.wikiart.org/en/paintings-by-style
### selecting japanese art

by_style = "https://www.wikiart.org/en/paintings-by-style"
url_style_soup = BeautifulSoup(url_r.urlopen(by_style), "html.parser")
all_styles_names = list(url_style_soup.find_all(class_="header"))

styles = []
for style in all_styles_names:
    extracted = re.sub(re.compile("<.*?>"), '', str(style)).strip()
    styles.append(extracted)

# Variable genre_to_scrape can be any from the list mentioned below:
"""
 'Ancient Greek Art',
 'Medieval Art',
 'Renaissance Art',
 'Post Renaissance Art',
 'Modern Art',
 'Contemporary Art',
 'Chinese Art',
 'Korean Art',
 'Japanese Art',
 'Islamic Art',
 'Native Art']
"""

style_to_scrape = "Japanese Art"

#################### Functions ####################

def get_urls_of_styles(style_to_scrape=style_to_scrape, styles=styles):
    check = style_to_scrape in styles
    assert check, "The style '{}' defined is not within the available styles: \n {}".format(style_to_scrape, styles)

    by_style = "https://www.wikiart.org/en/paintings-by-style"
    url_style_soup = BeautifulSoup(url_r.urlopen(by_style), "lxml")
    foo = url_style_soup.text
    style_to_scrape_idx = styles.index(style_to_scrape)
    if style_to_scrape_idx != len(styles):
        next_idx = style_to_scrape_idx + 1
    else:
        next_idx = None
    
    #get artist_names of specific styles
    foo_split = foo.split(styles[style_to_scrape_idx])[1].split(styles[next_idx])[0]
    foo_split = re.sub(r'[0-9]+', '', foo_split)
    substyle_names = foo_split.split("\n")
    substyle_names = [x.strip() for x in substyle_names if x]
    
    #extract urls from each name:
    a_tags = BeautifulSoup(str(url_style_soup.find_all(name=["li", "a"], class_="dottedItem")), "lxml").find_all("a")
    urls = [] 
    for a in a_tags:
        content = re.sub(r'[0-9]+', '', a.text).strip()
        if content in substyle_names:
            urls.append(a.attrs["href"])
            
    for i in range(len(urls)):
        urls[i] = "https://www.wikiart.org" + urls[i]
    
    urls = [url.replace("?select=featured", "") for url in urls]    
    return urls


def get_artist_url(genre = None, style_url = None):
    import re
    if genre is not None: 
        url_genre = "https://www.wikiart.org/en/paintings-by-genre/" + genre
        soup =  BeautifulSoup(url_r.urlopen(url_genre), "lxml")
        all_artists = soup.findAll("ul", {"class": "artists-group-list"})
        all_artists_txt = str(all_artists)
        all_artists_txt = all_artists_txt.replace('[<ul class="artists-group-list">', '')
        all_artists_txt = all_artists_txt.replace('</ul>]', '').strip()
        all_artists_txt = all_artists_txt.split("\n")
        no_artists = len(all_artists_txt)
        artist_url_list = [None] * no_artists
        for i in range(no_artists):
            url = re.split('\\bartistUrl=\\b', all_artists_txt[i])[1]
            url = url[0:url.find(" ")]
            #url = "https://www.wikiart.org/en/" + url
            artist_url_list[i] = url
    
    if style_url is not None:
        style_soup = BeautifulSoup(url_r.urlopen(style_url), "lxml")
        artist_urls = style_soup.find_all(class_="artists-list-container")
        artist_urls = BeautifulSoup(str(artist_urls), "lxml").find_all("li")
        full_name = []
        for artist in artist_urls:
            full_name.append(artist.text)
            
        artist_urls_converted = str(artist_urls).split(",")
        import re
        start = re.escape("artistUrl=")
        end   = re.escape(" ")
        artist_url_list = []
        for artist in artist_urls_converted:
            res = re.search('%s(.*)%s' % (start, end), artist)
            res = res.group(1)
            res = re.sub(r' .*', r'', res)
            artist_url_list.append(res)
        
    return artist_url_list

def get_all_images_from_artist_list(artist_url):
    artist_soup = BeautifulSoup(url_r.urlopen(artist_url), "lxml")
    href_to_img = BeautifulSoup(str(artist_soup.find_all("li", class_="painting-list-text-row")), "lxml").find_all("a")
    hrefs = []
    for href in href_to_img:
        hrefs.append(href["href"])
    
    hrefs = ["https://www.wikiart.org" + href for href in hrefs]
    
    image_url_list = [None] * len(hrefs)
    cnt = 0 
    for href in hrefs:
        ## Since this is in a loop add sleep to not have too many request
        time.sleep(1.5)
        img_soup = BeautifulSoup(url_r.urlopen(href), "lxml")
        img_extract = BeautifulSoup(str(img_soup.find_all("img", {"itemprop":"image"})), "lxml").find_all("img")
        for img in img_extract:
            image_url_list[cnt] = img["src"]#.replace("!Large.jpg", "")
            cnt += 1
        
    return image_url_list
    
def downloader(artist_url, genre_or_style):
    """
    Downloads images from each artist
    Input:
        artist_url: [str] i.e "wassily-kandinsky" will be concatenated as "https://www.wikiart.org/en/wassily-kandinsky"
        genre: passed in script beginning, genre_to_scrape. Needed for folder structure where images ought to be saved
    """
    print("Downloading from artist: {}".format(artist_url))
    
    ## important: add /all-works/text-list in order to really extract all images.
    ## otherwise the BS only scrapes 20 images (default) show on site. #ToDo use javascript functionality to show more images?!
    full_artist_url = "https://www.wikiart.org/en/" + artist_url + "/all-works/text-list"
    
    all_artist_images_full_url = get_all_images_from_artist_list(full_artist_url)
    n_images = len(all_artist_images_full_url)
    print("Trying to download {} images from artist {}".format(n_images, artist_url))
    j = 0 
    
    for img_url in all_artist_images_full_url:
        try:
            # try not to call site too often when downloading
            #https://www.codeproject.com/Questions/1060070/ConnectionResetError-WinError-An-existing-connecti
            time.sleep(random.random()*5.0)
            # add encode and decode to get rid of special characters if in link like 'danjûrô' becomes 'danjr'
            # use utf-8 encoding if you really want the full name
            path_to_be_saved = genre_or_style + "/" + artist_url.encode("ascii", "ignore").decode("ascii") + "_" + str(j) + ".jpg"
            #only download if file does not exist
            if not os.path.isfile(path_to_be_saved):
                image_on_web = url_r.urlopen(img_url.encode("utf-8", "ignore").decode("utf-8"))
                buf = image_on_web.read()
                downloaded_image = open(path_to_be_saved, "wb")
                downloaded_image.write(buf)
                downloaded_image.close()
                image_on_web.close()
            j += 1
            if j % 15 == 0:
                print("Downloaded number %d / %d from artist %s" % (j, n_images, artist_url))
        except Exception as e:
            print("failed downloading " + str(img_url))
            #traceback.print_exc()
            
    print("Finished downloading all images {} / {} from artist {}".format(n_images, n_images, artist_url))
    return None

def main_func(genre=None, style=None): 
    """ 
    Main python function
    Calls predefined functions and distributes the artists-images-to-be-downloaded onto ncores-1 jobs.
    """
    
    if not genre==None:
        print("Start image-downloading process from genre {}".format(genre))
        pool_of_threads = Pool(multiprocessing.cpu_count() - 1)
        url_list_genre = get_artist_url(genre = genre, style_url = None)
        #remove possible duplicates
        urls_list_genre = list(set(url_list_genre))
        pool_of_threads.starmap(downloader, zip(urls_list_genre, itertools.repeat(genre)))
        print("Finished image-downloading process from genre {}".format(genre))
        pool_of_threads.close()

        
    if not style==None:
        print("Start image-downloading process from style {}".format(style))
        pool_of_threads = Pool(multiprocessing.cpu_count() - 1)
        stlye_urls = get_urls_of_styles()
        urls_list_style = [get_artist_url(genre = None, style_url = url) for url in stlye_urls]
        urls_list_style = [val for sublist in urls_list_style for val in sublist]
        #remove possible duplicates 
        urls_list_style = list(set(urls_list_style))
        pool_of_threads.starmap(downloader, zip(urls_list_style, itertools.repeat(style)))	
        print("Finished image-downloading process from style {}".format(style))
        pool_of_threads.close()
        
    return None

### Start executions ###
os.chdir("../data")
## if user inserts
if len(sys.argv) >= 2:
    genre_to_scrape = sys.argv[1]
    style_to_scrape = sys.argv[2]
    
genres = [x.lower() for x in genres]
styles = [x.lower() for x in styles]
    
## in general check if either user input via shell command or manually insert in data_scraping.py is correct
if genre_to_scrape.lower() not in genres:
    print("Selected genre '{}' is not in provided wikiarts genre list: \n{}".format(genre_to_scrape, ", ".join(genres)))
if style_to_scrape.lower() not in styles:
    print("Selected style '{}' is not in provided wikiarts style list: \n{}".format(style_to_scrape, ", ".join(styles)))         
             
## otherwise take genre_to_scrape and style_to_scrape from script

##genre
if not os.path.exists(os.path.join(os.getcwd(), genre_to_scrape)):
    os.mkdir(genre_to_scrape)
   
print("Building a list of all the paintings to be downloaded from genre: " + genre_to_scrape + "\n This might take a while...")
## Call main python function
main_func(genre = genre_to_scrape, style = None)
print("Downloaded images are at: " + os.path.join(os.getcwd(), genre_to_scrape))
  
##style
if not os.path.exists(os.path.join(os.getcwd(),style_to_scrape)):
    os.mkdir(style_to_scrape)
        
print("Building a list of all the paintings to be downloaded from style: " + style_to_scrape + "\n This might take a while...")
## Call main python function
main_func(genre = None, style = style_to_scrape)
print("Downloaded images are at: " + os.path.join(os.getcwd(), style_to_scrape))

 
## END ##