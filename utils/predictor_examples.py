import os
import numpy as np
import random

from refvg_loader import RefVGLoader
from data_transfer import boxes_to_mask, polygons_to_mask
from visualize_utils import save_pred_to_png


def box_rand_predictor(split='val', eval_img_count=5, out_path='output/baselines/box_rand/predictions'):
    """
    randomly generate a box on the image as the predicted mask.
    Save predictions as binary PNG images into the out_path.
    """
    print('start of box_rand_predictor')
    loader = RefVGLoader(split=split, input_anno_only=True)
    if eval_img_count < 0:
        eval_img_count = len(loader.img_ids)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for img_i, img_id in enumerate(loader.img_ids):
        print('box_rand_predictor: img %d / %d' % (img_i, eval_img_count))
        img_data = loader.get_img_ref_data(img_id)
        for task_i, task_id in enumerate(img_data['task_ids']):
            w = img_data['width']
            h = img_data['height']
            x1 = random.randint(0, h - 1)
            x2 = random.randint(0, h - 1)
            if x1 > x2:
                t = x1
                x1 = x2
                x2 = t
            y1 = random.randint(0, w - 1)
            y2 = random.randint(0, w - 1)
            if y1 > y2:
                t = y1
                y1 = y2
                y2 = t
            pred_mask = boxes_to_mask([[x1, y1, x2, y2]], w, h, xywh=False)
            fpath = os.path.join(out_path, '%s.png' % task_id)
            save_pred_to_png(pred_mask, fpath)

        if img_i >= eval_img_count > 0:
            break

    print('box_rand_predictor Done!')
    return


def vg_gt_predictor(split='val', eval_img_count=-1, out_path=None):
    """
    vg boxes used to generate the phrase as the predicted mask.
    Save prediction results to a dict.
    Note that it's not a valid predictor to compare on the dataset, because it requires vg_gt boxes,
    which is not included in the input annotations.
    """
    loader = RefVGLoader(split=split, include_vg_scene_graph=True, input_anno_only=False)
    if eval_img_count < 0:
        eval_img_count = len(loader.img_ids)
    predictions = dict()
    for img_i, img_id in enumerate(loader.img_ids):
        print('vg_predictor: img %d / %d' % (img_i, eval_img_count))
        img_data = loader.get_img_ref_data(img_id)
        predictions[img_id] = dict()
        for task_i, task_id in enumerate(img_data['task_ids']):
            pred_box_list = img_data['vg_boxes'][task_i]
            pred_mask = boxes_to_mask(pred_box_list, img_data['width'], img_data['height'], xywh=True)
            pred_mask = np.packbits(pred_mask.astype(np.bool))
            predictions[img_id][task_id] = {'pred_boxlist': pred_box_list, 'pred_mask': pred_mask}
        if len(predictions) >= eval_img_count > 0:
            break

    if out_path is not None:
        print('rang_vg_predictor: saving predictions to %s' % out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        fname = split
        if eval_img_count > 0:
            fname += '_%d' % eval_img_count
        fname += '.npy'
        f_path = os.path.join(out_path, fname)
        np.save(f_path, predictions)
    print('vg_predictor Done!')
    return predictions


def vg_rand_predictor(split='val', eval_img_count=-1, out_path=None):
    """
    randomly pick one vg box as the predicted mask.
    Save prediction results to a dict.
    Note that it's not a valid predictor to compare on the dataset, because it requires vg boxes,
    which is not included in the input annotations.
    """
    loader = RefVGLoader(split=split, include_vg_scene_graph=True)
    if eval_img_count < 0:
        eval_img_count = len(loader.img_ids)
    predictions = dict()
    for img_i, img_id in enumerate(loader.img_ids):
        print('rand_vg_predictor: img %d / %d' % (img_i, eval_img_count))
        img_data = loader.get_img_ref_data(img_id)
        predictions[img_id] = dict()
        for task_i, task_id in enumerate(img_data['task_ids']):
            pred_boxes = [random.choice(img_data['img_vg_boxes'])]
            pred_mask = boxes_to_mask(pred_boxes, img_data['width'], img_data['height'], xywh=True)
            pred_mask = np.packbits(pred_mask.astype(np.bool))
            predictions[img_id][task_id] = {'pred_boxlist': pred_boxes, 'pred_mask': pred_mask}
        if len(predictions) >= eval_img_count > 0:
            break

    if out_path is not None:
        print('rang_vg_predictor: saving predictions to %s' % out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        fname = split
        if eval_img_count > 0:
            fname += '_%d' % eval_img_count
        fname += '.npy'
        f_path = os.path.join(out_path, fname)
        np.save(f_path, predictions)
    print('rand_vg_predictor Done!')
    return predictions


def ins_rand_predictor(split='val', eval_img_count=-1, out_path=None):
    """
    randomly pick one instance mask as the predicted mask.
    Save prediction results to a dict.
    Note that it's not a valid predictor to compare on the dataset, because it requires instances,
    which is not included in the input annotations.
    """
    loader = RefVGLoader(split=split)
    if eval_img_count < 0:
        eval_img_count = len(loader.img_ids)
    predictions = dict()
    for img_i, img_id in enumerate(loader.img_ids):
        print('ins_rand_predictor: img %d / %d' % (img_i, eval_img_count))
        img_data = loader.get_img_ref_data(img_id)
        predictions[img_id] = dict()
        for task_i, task_id in enumerate(img_data['task_ids']):
            pred_ix = random.choice(range(len(img_data['img_ins_boxes'])))
            pred_box = img_data['img_ins_boxes'][pred_ix]
            pred_polygons = img_data['img_ins_Polygons'][pred_ix]
            pred_mask = polygons_to_mask(pred_polygons, img_data['width'], img_data['height'])
            correct = 0
            if pred_box in img_data['gt_boxes']:
                correct = 1
            pred_mask = np.packbits(pred_mask.astype(np.bool))
            predictions[img_id][task_id] = {'pred_boxlist': [pred_box], 'pred_mask': pred_mask, 'correct': correct}
        if len(predictions) >= eval_img_count > 0:
            break

    if out_path is not None:
        print('ins_rand_predictor: saving predictions to: %s' % out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        fname = split
        if eval_img_count > 0:
            fname += '_%d' % eval_img_count
        fname += '.npy'
        f_path = os.path.join(out_path, fname)
        np.save(f_path, predictions)
    print('ins_rand_predictor Done!')
    return predictions
