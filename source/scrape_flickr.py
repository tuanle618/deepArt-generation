import flickrapi
import urllib
import os

API_key = "079921da9a005d9590836da31229f7f2"
API_secret_key = "62ab87626c9b4fb3"
keywords = ["nude painting", "nude paintings"]

def crawl_flickr(keyword, limit=10000):
    # Flickr api access key
    flickr = flickrapi.FlickrAPI(API_key, API_secret_key, cache=False)
    print("Crawl keyword: {}".format(keyword))
    photos = flickr.walk(text=keyword,
                         tag_mode="all",
                         tags=keyword,
                         extras="url_c",
                         per_page=400,
                         sort="relevance")
    urls = [None] * limit
    for i, photo in enumerate(photos):

        if i > limit:
            break
        urls[i] = photo.get("url_c")

        ## print out every 1000 urls information
        if i % 1000 == 0 and i != 0:
            print("Retrieved url number {}".format(i))

    urls = [url for url in urls if url is not None]

    return urls


def download_image(url, filename):
    urllib.urlretrieve(url, filename)


all_urls = [crawl_flickr(keyword, limit=10000) for keyword in keywords]
all_urls = [item for sublist in all_urls for item in sublist]
len(all_urls)