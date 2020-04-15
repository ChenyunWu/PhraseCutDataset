import os
import numpy as np
import json
from tqdm import tqdm

from PhraseCutDataset.utils.file_paths import img_info_fpath
from PhraseCutDataset.utils.visualize_utils import save_pred_to_png


def pred_dict_to_png(dict_path):
    out_png_path = os.path.join(os.path.dirname(dict_path), os.path.basename(dict_path).split('.')[0] + '_pngs')
    print(out_png_path)
    if not os.path.exists(out_png_path):
        os.makedirs(out_png_path)
    with open(img_info_fpath, 'r') as f:
        imgs_info = json.load(f)
        ImgInfo = {img['image_id']: img for img in imgs_info}

    pred_dict = np.load(dict_path, allow_pickle=True, encoding='bytes').item()
    print(len(pred_dict))
    none_count = 0
    for img_id, img_pred in tqdm(pred_dict.items(), desc='img_pred to png files'):
        if img_id not in ImgInfo:
            continue
        img_data = ImgInfo[img_id]
        for task_id, task_pred in img_pred.items():
            pred_mask_bin = task_pred.get(b'pred_mask', None)
            if pred_mask_bin is None:
                pred_mask_bin = task_pred.get('pred_mask', None)
            if pred_mask_bin is None:
                none_count += 1
                print('WARNING %d: No pred_mask' % none_count, task_id, task_pred.keys())
                # pred_mask = np.zeros((img_data['height'], img_data['width']))
            else:
                pred_mask = np.unpackbits(pred_mask_bin)[:img_data['height'] * img_data['width']]\
                    .reshape((img_data['height'], img_data['width']))
                file_path = os.path.join(out_png_path, '%s.png' % task_id)
                save_pred_to_png(pred_mask, file_path)


if __name__ == '__main__':
    for top_k in [1, 15]:
        pred_dict_to_png('/home/chenyun/work1/PhraseCutEnsemble/output/eval_refvg/'
                         'det_mattnet_pred_0.01_%d_cat_test0/test_2937.npy' % top_k)
    # pred_dict_to_png('/home/chenyun/work1/PhraseCutEnsemble/output/eval_refvg/'
    #                  'det_mattnet_pred_0.15_50_det_test0/test_2937.npy')
    # pred_dict_to_png('/home/chenyun/work1/PhraseCutEnsemble/output/eval_refvg/'
    #                  'det_mattnet_pred_0.01_50_det_test0/test_2937.npy')
    # pred_dict_to_png('/home/chenyun/work1/PhraseCutEnsemble/output/eval_refvg/'
    #                  'rmi_pred_test0/test.npy')
