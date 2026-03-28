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
        'name_att_rel_count': 'https://drive.google.com/uc?id=10lWrt2Oy7Im-QofRSNKNu9qa_DS0IcbR',
        'img_info': 'https://drive.google.com/uc?id=1xB9eExJo35K3DQu8PnBWZ1q_OMoMl8-i',
        'refer_miniv': 'https://drive.google.com/uc?id=1oLcTQ1blTIQyu5ZMelQN2HSniuQaxM4E',
        'refer_test': 'https://drive.google.com/uc?id=1jrzXm1gcq6f5hNDeamZd0UmyyHUv61IZ',
        'refer_train': 'https://drive.google.com/uc?id=1qx-0q6r9r0YUGpoyT0B8HJKmUFWQDSu7',
        'refer_val': 'https://drive.google.com/uc?id=1UyojArOFPlsSeNbA9fHWjCjOOU-OCohG',
        'refer_input_miniv': 'https://drive.google.com/uc?id=1QPZ35eSLRRczM4OMjlkiI88m7JGS5rsN',
        'refer_input_test': 'https://drive.google.com/uc?id=1xfr3MKPSSPUfIf3i_LQA3JAEdbCRyMiD',
        'refer_input_train': 'https://drive.google.com/uc?id=1udrL3DM6Ksml7jXGRY8PSPSCk5sGIsGd',
        'refer_input_val': 'https://drive.google.com/uc?id=1DjJRoTdJGpee8k4QKfjhV97XQLjuiLk0',

        'scene_graphs_train': 'https://drive.google.com/uc?id=14Gjf8YA8ryw7VZXoQkHc3eiD-3mRHuJS',
        'scene_graphs_val': 'https://drive.google.com/uc?id=1X0bU8TuE8o_yn4lFxA3DvHSKwn_esbPu',
        'scene_graphs_test': 'https://drive.google.com/uc?id=1KG5D_Ah88b-rC-gRt1nJ_pXgET9aUt0c',
        'scene_graphs_miniv': 'https://drive.google.com/uc?id=1xT10sTv8S7LD9b5ZeSZvVxOVbpJkBECm',
        'skip': 'https://drive.google.com/uc?id=1pqciUIP2OgewoRGW2AsoT2P0Bb20KCpT'
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
