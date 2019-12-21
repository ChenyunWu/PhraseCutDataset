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
        'name_att_rel_count': 'https://drive.google.com/uc?id=1iDmxDa3Aj_c47mZmvVTyeGq4pCv7EWj_',
        'img_info': 'https://drive.google.com/uc?id=1BDWBLV1FdKNnCtTX4hIlnKrqc3KOKp6X',
        'refer_miniv': 'https://drive.google.com/uc?id=1S9vJl3KnE1uySH-bCYfC8bq1BWz-a1Ny',
        'refer_test': 'https://drive.google.com/uc?id=17JW1yfAF43M5Q2CVPxSoPIfOsUjU7xHP',
        'refer_train': 'https://drive.google.com/uc?id=1yuQz4S5ON7kf9TZa9tFnMjMFqwOlTomv',
        'refer_val': 'https://drive.google.com/uc?id=1J5GFEhN3z0_LiGOI8tQSRLSwsq4ph9RF',
        'scene_graphs_train': 'https://drive.google.com/uc?id=1oGkrOyDgGzkJlA4Ea1MiU0hBbROu-kZx',
        'scene_graphs_val': 'https://drive.google.com/uc?id=1su1szG3UZhLLChl9VYifJ3HTrxSJBA5A',
        'scene_graphs_test': 'https://drive.google.com/uc?id=1MdsoNtIkc-vyHGuKlG0XrMGCe8HGEIaY',
        'scene_graphs_miniv': 'https://drive.google.com/uc?id=1ioYj5Lq-a2m0yaaV7wcPVVfv-npNJXbZ',
        'skip': 'https://drive.google.com/uc?id=16bil5jCmDXJz5hGKTypWwiBPjRRSqetA'
    }

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)


    gdown.download(urls['name_att_rel_count'], str(name_att_rel_count_fpath), quiet=False)
    gdown.download(urls['img_info'], str(img_info_fpath), quiet=False)

    if download_skip:
        gdown.download(urls['skip'], str(skip_fpath), quiet=False)

    for s in splits:
        if download_refer:
            gdown.download(urls['refer_%s' % s], str(refer_fpaths[s]), quiet=False)
        if download_sg:
            gdown.download(urls['scene_graphs_%s' % s], str(vg_scene_graph_fpaths[s]), quiet=False)
    return


def download_images(splits):
    if not os.path.exists(img_fpath):
        os.makedirs(img_fpath)

    to_download = dict()
    info = json.load(open(img_info_fpath))
    for img in info:
        if img['split'] in splits:
            to_download['%d.jpg' % img['image_id']] = img['url']
    print('%d imgs to be downloaded' % len(to_download))
    c = 0
    for fname, url in to_download.items():
        download(url, os.path.join(img_fpath, fname))
        c += 1
        if c % 100 == 0:
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
    download_images(splits)
    return


if __name__ == '__main__':
    main()
