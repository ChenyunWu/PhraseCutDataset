import os
import numpy as np
import random

from visualize_utils import save_pred_to_png
from refvg_loader import RefVGLoader
from data_transfer import boxes_to_mask, polygons_to_mask, xyxy_to_xywh


def box_rand_pred(img_data, task_i):
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

    pred_boxes = xyxy_to_xywh([[x1, y1, x2, y2]])
    pred_mask = boxes_to_mask(pred_boxes, w, h, xywh=True)
    return pred_mask, pred_boxes, None


def vg_gt_pred(img_data, task_i):
    pred_boxes = img_data['vg_boxes'][task_i]
    pred_mask = boxes_to_mask(pred_boxes, img_data['width'], img_data['height'], xywh=True)
    return pred_mask, pred_boxes, None


def vg_rand_pred(img_data, task_i):
    pred_boxes = [random.choice(img_data['img_vg_boxes'])]
    pred_mask = boxes_to_mask(pred_boxes, img_data['width'], img_data['height'], xywh=True)
    return pred_mask, pred_boxes, None


def ins_rand_pred(img_data, task_i):
    pred_ix = random.choice(range(len(img_data['img_ins_boxes'])))
    pred_box = img_data['img_ins_boxes'][pred_ix]
    pred_polygons = img_data['img_ins_Polygons'][pred_ix]
    pred_mask = polygons_to_mask(pred_polygons, img_data['width'], img_data['height'])
    correct = 0
    if pred_box in img_data['gt_boxes']:
        correct = 1
    return pred_mask, [pred_box], correct


pred_func_fetcher = {'box_rand': box_rand_pred,
                     'vg_gt': vg_gt_pred,
                     'vg_rand': vg_rand_pred,
                     'ins_rand': ins_rand_pred}


def example_predictor(refvg_loader=None, split='val', eval_img_count=-1, pred_method_name='box_rand',
                      out_png_path='output/baselines/box_rand/predictions', out_dict_path=None):
    """
    Select the pred method by pred_method_name, make predictions on 'split'.
    Note that 'vg_gt', 'vg_rand' and 'ins_rand' are not rigid predictors to compare on the dataset,
    because they require extra annotations not included as the valid input.
    Save predictions as binary PNG images into the out_png_path.
    [Obsolete:] Save all predictions in a dict as a npy file to out_dict_path. Only when out_png_path is None.
    """
    # Obsolete
    if out_dict_path is not None and out_png_path is None:
        return example_predictor_obsolete(refvg_loader=refvg_loader, split=split,
                                          eval_img_count=eval_img_count, pred_method_name=pred_method_name,
                                          out_dict_path=out_dict_path)

    print('start of example predictor:', pred_method_name)
    loader = refvg_loader
    if loader is None:
        if pred_method_name == 'box_rand':
            # this is how you should create the loader for prediction
            loader = RefVGLoader(split=split, input_anno_only=True)
        elif pred_method_name == 'ins_rand':
            loader = RefVGLoader(split=split, input_anno_only=False)
        elif pred_method_name in ['vg_gt', 'vg_rand']:
            loader = RefVGLoader(split=split, include_vg_scene_graph=True)
    if eval_img_count < 0:
        eval_img_count = len(loader.img_ids)

    pred_func = pred_func_fetcher[pred_method_name]
    # prepare path
    if out_png_path is not None and not os.path.exists(out_png_path):
        os.makedirs(out_png_path)

    for img_i, img_id in enumerate(loader.img_ids):
        print('predicting on: img %d / %d' % (img_i, eval_img_count))
        img_data = loader.get_img_ref_data(img_id)
        for task_i, task_id in enumerate(img_data['task_ids']):
            # make prediction
            pred_mask, pred_boxlist, correct = pred_func(img_data, task_i)
            # save results
            if out_png_path is not None:
                file_path = os.path.join(out_png_path, '%s.png' % task_id)
                save_pred_to_png(pred_mask, file_path)

        if img_i >= eval_img_count > 0:
            break

    print('example predictor %s Done!' % pred_method_name)
    return None


def example_predictor_obsolete(refvg_loader=None, split='val', eval_img_count=-1, pred_method_name='box_rand',
                               out_dict_path=None):
    """
    Save all predictions in a dict as a npy file to out_dict_path.
    """
    loader = refvg_loader
    if loader is None:
        if pred_method_name == 'box_rand':
            # this is how you should create the loader for prediction
            loader = RefVGLoader(split=split, input_anno_only=True)
        elif pred_method_name == 'ins_rand':
            loader = RefVGLoader(split=split, input_anno_only=False)
        elif pred_method_name in ['vg_gt', 'vg_rand']:
            loader = RefVGLoader(split=split, include_vg_scene_graph=True)
    if eval_img_count < 0:
        eval_img_count = len(loader.img_ids)

    pred_func = pred_func_fetcher[pred_method_name]

    predictions = dict()

    for img_i, img_id in enumerate(loader.img_ids):
        print('predicting on: img %d / %d' % (img_i, eval_img_count))
        img_data = loader.get_img_ref_data(img_id)

        predictions[img_id] = dict()

        for task_i, task_id in enumerate(img_data['task_ids']):
            # make prediction
            pred_mask, pred_boxlist, correct = pred_func(img_data, task_i)
            # save results
            if out_dict_path is not None:
                pred_mask = np.packbits(pred_mask.astype(np.bool))
                predictions[img_id][task_id] = {'pred_boxlist': pred_boxlist, 'pred_mask': pred_mask}

        if img_i >= eval_img_count > 0:
            break

    print('example predictor %s: saving predictions to %s' % (pred_method_name, out_dict_path))
    if not os.path.exists(out_dict_path):
        os.makedirs(out_dict_path)
    fname = split
    if eval_img_count > 0:
        fname += '_%d' % eval_img_count
    fname += '.npy'
    f_path = os.path.join(out_dict_path, fname)
    np.save(f_path, predictions)

    print('example predictor %s Done!' % pred_method_name)
    return predictions
