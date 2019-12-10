import os
import json
import argparse
import requests

from utils.file_paths import img_fpath, img_info_fpath


def download(url, fpath):
    myfile = requests.get(url, allow_redirects=True)
    open(fpath, 'wb').write(myfile.content)


def download_images(path=None, splits='train_val_test_miniv'):
    if splits is None or len(splits) == 0:
        splits = 'train_val_test_miniv'
    splits = splits.split('_')
    if path is None or len(path) == 0:
        path = img_fpath
    if not os.path.exists(path):
        os.makedirs(path)

    to_download = dict()
    info = json.load(open(img_info_fpath))
    for img in info:
        if img['split'] in splits:
            to_download['%d.jpg' % img['image_id']] = img['url']
    print('%d imgs to be downloaded' % len(to_download))
    c = 0
    for fname, url in to_download.items():
        download(url, os.path.join(path, fname))
        c += 1
        if c % 100 == 0:
            print('doloaded %d / %d' % (c, len(to_download)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default='',
                        help='dataset split to evaluate: val, miniv, test, train, val_miniv, etc')
    parser.add_argument('-p', '--path', type=str, default=None, help='output path for images')
    args = parser.parse_args()

    download_images(args.path, args.split)
    return


if __name__ == '__main__':
    main()
