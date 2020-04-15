import os
import json
import argparse
import requests
import gdown

from utils.file_paths import *


def download(url, fpath):
    myfile = requests.get(url, allow_redirects=True)
    open(fpath, 'wb').write(myfile.content)


def download_annotations(splits, download_refer=True, download_sg=False, download_skip=False):
    urls = {
        'name_att_rel_count': 'https://drive.google.com/uc?id=1QbunEpB0l6PXCTVR5L7YsJM7-eNoKjHM',
        'img_info': 'https://drive.google.com/uc?id=1UmjLJx9BGE9ruOXK1T4iGCYKurY0jNOL',
        'refer_miniv': 'https://drive.google.com/uc?id=1HL9YX8rmMTAxAblkBAVv2PO4aNHfFVDe',
        'refer_test': 'https://drive.google.com/uc?id=1o1waJid-D5EvyIoDUyqUbTYofcIPcLZt',
        'refer_train': 'https://drive.google.com/uc?id=1VCFRajJ4YXPmW5SJg6lS0uKaVAysO61F',
        'refer_val': 'https://drive.google.com/uc?id=1Q2HFlss5Y2zLjTydQMWzq6u0iFsGPNnd',
        'refer_input_miniv': 'https://drive.google.com/uc?id=1aFjegXv6VFgbDdcKwB7S4whSoagfN3pr',
        'refer_input_test': 'https://drive.google.com/uc?id=1oKPI3pAGL36iELIbZ8xBCRdv8XFjNGyY',
        'refer_input_train': 'https://drive.google.com/uc?id=1b59w_IcfpvNfBhraSKw5RfmFHFJ4lIAb',
        'refer_input_val': 'https://drive.google.com/uc?id=1-t1Qha7Bu9DKwFxUL9Tt6A6nw6ANtImI',
        'scene_graphs_train': 'https://drive.google.com/uc?id=1qsLfaQ4uBUk2wn0BBbkQtofPJ2LjK08C',
        'scene_graphs_val': 'https://drive.google.com/uc?id=1wbsUTxKHjA2dqjAeF63XQvF_LqNFQi1W',
        'scene_graphs_test': 'https://drive.google.com/uc?id=1z2ll6QKPYHTCv4MKUKEmr7xVkgkgDaOR',
        'scene_graphs_miniv': 'https://drive.google.com/uc?id=1p9KT6cw8gP6NkPdCy5Hp4X8_h4esBwKL',
        'skip': 'https://drive.google.com/uc?id=1L7j_-9lGk90BuqiQR15W95VQ8frejAfe'
    }

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    gdown.download(urls['name_att_rel_count'], str(name_att_rel_count_fpath), quiet=False, proxy=None)
    gdown.download(urls['img_info'], str(img_info_fpath), quiet=False, proxy=None)

    if download_skip:
        gdown.download(urls['skip'], str(skip_fpath), quiet=False, proxy=None)

    for s in splits:
        if download_refer:
            gdown.download(urls['refer_%s' % s], str(refer_fpaths[s]), quiet=False, proxy=None)
            gdown.download(urls['refer_input_%s' % s], str(refer_input_fpaths[s]), quiet=False, proxy=None)
        if download_sg:
            gdown.download(urls['scene_graphs_%s' % s], str(vg_scene_graph_fpaths[s]), quiet=False, proxy=None)
    return


def download_images(splits):
    if not os.path.exists(img_fpath):
        os.makedirs(img_fpath)

    to_download = dict()
    info = json.load(open(img_info_fpath))
    for img in info:
        if img['split'] in splits:
            to_download['%d.jpg' % img['image_id']] = img['url']
    print('%d imgs to be downloaded. This may take some time.' % len(to_download))
    c = 0
    for fname, url in to_download.items():
        download(url, os.path.join(img_fpath, fname))
        c += 1
        d = min(len(to_download) / 10, 20)
        if c % d == 0:
            print('downloaded %d / %d images' % (c, len(to_download)))
    print('Finished downloading images.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default='train_val_test_miniv',
                        help='dataset split to download: val, miniv, test, train, val_miniv, ...')
    parser.add_argument('--download_refer', type=int, default=1, help='Whether to download the annotations (0 / 1)')
    parser.add_argument('--download_img', type=int, default=1, help='Whether to download the images or not (0 / 1)')
    parser.add_argument('--download_graph', type=int, default=0, help='Whether to download VG scene graph (0 / 1)')
    parser.add_argument('--download_skip', type=int, default=0, help='Whether to download skipped phrases (0 / 1)')
    args = parser.parse_args()

    splits = args.split
    if splits is None or len(splits) == 0:
        splits = 'train_val_test_miniv'
    splits = splits.split('_')

    download_annotations(splits, download_refer=args.download_refer, download_sg=args.download_graph,
                         download_skip=args.download_skip)
    if args.download_img:
        download_images(splits)
    return


if __name__ == '__main__':
    main()
