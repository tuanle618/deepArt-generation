import flickrapi
import urllib.request
import os
import argparse
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

def args_parser():
    parser = argparse.ArgumentParser(description="Scrape flickr for images based on keyword")

    parser.add_argument("-kw", "--keywords", type=str, default="nude painting", dest="keywords", nargs= "+",
                        help="String which keywords to scrape. If more keywords should be scraped use this parameter several times.\
                        e.g --keyword 'nude art' --keyword 'landscape'. Default: 'nude painting'")
    parser.add_argument("-ac", "--api_cred", type=str, dest="api_cred", default=None,
                        help="Where .txt file for flickr API is stored. Note it should be 2 lines\
                        with API_KEY=... and API_SECRET_KEY=... Default: None and checks if its inserted within this script.")
    parser.add_argument("-sp", "--save_path", type=str, dest="save_path", default="nude-painting",
                        help="Directory where the images should be saved. The images will be saved to 'data/save_path' \
                        subdirectory where <save_path> needs to be defined. Default 'nude-painting'")
    parser.add_argument("-l", "--limit", type=int, dest="limit", default=30000,
                        help="Maximum number of images to scrape from flickr. Default: 30000")
    parser.add_argument("-nw", "--n_workers", type=int, default=4, dest="n_workers",
                        help="For downloading images split the download into <n_workers> processes. Default: 4")

    args = parser.parse_args()

    return args

def load_api_cred(path):
    """
    Loads the API_KEY and API_SECRET_KEY from a .txt or .json file.
    :param path: Path where keys are stored
    :return: API_KEY and API_SECRECT_KEY as string
    """
    if path.endswith("txt"):
        with open(path, "r") as fp:
            api_creds = fp.readlines()
            api_creds = [foo.replace("\n", "") for foo in api_creds]

        ## check if length is 2
        if len(api_creds) != 2:
            print("More Lines within {} file. Check again!".format(path))
            quit()

        a = api_creds[0]
        b = api_creds[1]

        a = a.split("=")
        b = b.split("=")

        if a[0].lower() == "api_key":
            API_KEY = a[1]
        elif a[0].lower() == "api_secret_key":
            API_SECRET_KEY = a[1]
        else:
            print("Wrong naming with variable {}".format(a[0]))
            quit()

        if b[0].lower() == "api_key":
            API_KEY = b[1]
        elif b[0].lower() == "api_secret_key":
            API_SECRET_KEY = b[1]
        else:
            print("Wrong naming with variable {}".format(b[0]))
            quit()

    elif path.endswith("json"):
        import json
        with open(path, "r") as fp:
            api_creds = json.load(fp)

        for key, value in api_creds.items():
            if key.lower() == "api_key":
                API_KEY = value
            elif key.lower() == "api_secret_key":
                API_SECRET_KEY = value
            else:
                print("JSON file not containing elements API_KEY and/or API_SECRET_KEY! Check again")
                quit()

    return str(API_KEY), str(API_SECRET_KEY)

def crawl_flickr(keyword, limit, flickr):
    """
    Crawls the flickr API based on a keyword and retrieves maximum limit urls.
    :param keyword: Which text should be searched
    :param limit: Maximal number of images to be scraped. Note that scraped images can be less if flickr does not provide as many images as wished.
    :param flickr: flickr.FLICKRAPI instace
    :return: list of image urls
    """
    print("\nCrawl keyword: {}".format(keyword))
    #https://www.flickr.com/services/api/flickr.photos.search.html
    photos = flickr.walk(text=keyword,
                         tag_mode="all",
                         extras="url_c",
                         per_page=500,
                         sort="relevance")
    urls = [None] * limit
    for i, photo in tqdm(enumerate(photos), total=limit):

        if i == limit:
            break

        urls[i] = photo.get("url_c")
        ## print out every 1000 urls information
        if i % 1000 == 0 and i != 0 or i == limit-1:
            tqdm.write("Retrieved url number {}/{}".format(i+1 if i == limit-1 else i, limit))

    urls = [url for url in urls if url is not None]

    return urls

def download_image(tuple):
    filename, url = tuple
    try:
        urllib.request.urlretrieve(url, filename)
        return 1
    except:
        return 0

if __name__ == '__main__':

    print("Python script to scrape images from flickr and download images.")
    args = args_parser()
    print(args)

    keywords = args.keywords
    limit = args.limit
    save_path = "../data/" + args.save_path
    n_workers = args.n_workers
    api_cred = args.api_cred

    if api_cred is not None:
        API_KEY, API_SECRET_KEY = load_api_cred(path=api_cred)
    else:
        ## Insert Flickr API credentials here:
        API_KEY = None
        API_SECRET_KEY = None

    # Flickr api access key
    flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET_KEY, cache=True)

    if api_cred is None and API_KEY is None and API_SECRET_KEY is None:
        print("No API credentials where inserted as argparse credentials path or within this script! Check again")
        quit()

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    ## Scrape links:
    if isinstance(keywords, list) and len(keywords) > 1:
        all_urls = [crawl_flickr(keyword, limit, flickr) for keyword in keywords]
        all_urls = [item for sublist in all_urls for item in sublist]
    else:
        all_urls = crawl_flickr(keywords, limit, flickr)

    print("\nRetrieved {} urls out of {}".format(len(all_urls), limit))
    print("Removing duplicate URLs..")
    ## remove duplicates
    all_urls = list(set(all_urls))
    print("Total number of scraped links for keywords {} is: {}".format(str(keywords), len(all_urls)))

    ## Download images:
    # Preprare tupled list to pass over to multithread
    all_urls = [("{}/{}.jpg".format(save_path, i), url) for i, url in enumerate(all_urls)]
    results = ThreadPool(n_workers).imap_unordered(func=download_image, iterable=all_urls)

    n_sucess = 0
    for sucess in tqdm(results, total=len(all_urls)):
        n_sucess += sucess
        ## print out every 1000
        if n_sucess % 1000 == 0 and n_sucess != 0 or n_sucess == len(all_urls):
            tqdm.write("Downloaded {}/{}".format(n_sucess, len(all_urls)))

    print("Downloaded {} images and stored at {}".format(n_sucess, save_path))