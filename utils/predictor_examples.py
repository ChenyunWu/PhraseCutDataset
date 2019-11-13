from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random

current_path = os.path.realpath(__file__)
dataset_path = os.path.join(current_path, '../..')
sys.path.append(os.path.abspath(dataset_path))

from utils.refvg_loader import RefVGLoader
from utils.data_transfer import *


def vg_gt_predictor(split='val', eval_img_count=-1, out_path='output/eval_refvg/vg_gt'):
    """vg boxes used to generate the phrase as the predicted mask"""
    loader = RefVGLoader(split=split)
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


def vg_rand_predictor(split='val', eval_img_count=-1, out_path='output/eval_refvg/vg_rand'):
    """randomly pick one vg box as the predicted mask"""
    loader = RefVGLoader(split=split)
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


def ins_rand_predictor(split='val', eval_img_count=-1, out_path='output/eval_refvg/ins_rand'):
    """randomly pick one vg box as the predicted mask"""
    loader = RefVGLoader(split=split)
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
