from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('agg')
current_path = os.path.realpath(__file__)
dataset_path = os.path.join(current_path, '..')
sys.path.append(os.path.abspath(dataset_path))

import utils.subset as subset_utils
from utils.visualize_utils import gt_visualize_to_file, pred_visualize_to_file
from utils.refvg_loader import RefVGLoader


def visualize(pred_eval_dict, out_path, gt_plot_path='data/refvg/visualizations', refvg_loader=None, refvg_split=None,
              all_task_num=200, subset_task_num=0, verbose=True):
    # prepare paths
    html_path = os.path.join(out_path, 'htmls')
    if not os.path.exists(html_path):
        os.makedirs(html_path)
    path_dict = dict()
    for exp_name, pred_eval in pred_eval_dict.items():
        exp_path = os.path.join(out_path, exp_name)
        result_path = os.path.join(exp_path, 'results.txt')
        pred_plot_path = os.path.join(exp_path, 'pred_plots')
        path_dict[exp_name] = [result_path, pred_plot_path]
        if not os.path.exists(pred_plot_path):
            os.makedirs(pred_plot_path)
        if not os.path.exists(gt_plot_path):
            os.makedirs(gt_plot_path)

    if refvg_loader is None:
        refvg_loader = RefVGLoader(split=refvg_split)

    if subset_task_num > 0:
        subsets = subset_utils.subsets
    else:
        subsets = ['all']
    subset_dict = dict()
    for s in subsets:
        subset_dict[s] = list()
    for img_id, img_pred in pred_eval_dict.values()[0].items():  # assert all predictions consist same tasks
        for task_id, pred in img_pred.items():
            assert 'subsets' in pred
            for subset in pred['subsets']:
                subset_dict[subset].append((img_id, task_id))

    # generate html for each subset
    for subset, img_task_ids in subset_dict.items():
        # sample
        if subset == 'all':
            sample_num = all_task_num
        else:
            sample_num = subset_task_num
        total_num = len(img_task_ids)
        if len(img_task_ids) > sample_num:
            img_task_ids = random.sample(img_task_ids, sample_num)
        if verbose:
            print('generating plots for %s(%d)' % (subset, len(img_task_ids)))
        # generate plots (if not already there)
        for i, (img_id, task_id) in enumerate(img_task_ids):
            img_data = refvg_loader.get_img_ref_data(img_id)
            for exp_name, pred_eval in pred_eval_dict:
                pred_mask = pred_eval[img_id][task_id]['pred_mask']
                pred_mask = np.unpackbits(pred_mask)[:img_data['height'] * img_data['width']] \
                    .reshape((img_data['height'], img_data['width']))
                g1 = gt_visualize_to_file(img_data, task_id, out_path=gt_plot_path)
                g2 = pred_visualize_to_file(img_data, task_id, pred_mask=pred_mask, out_path=path_dict[exp_name][1])
                if verbose:
                    print('img %d exp %s: gt plot - %s, pred plot - %s' % (i, exp_name, g1, g2))

        # html parameters
        gt_rel_path = os.path.relpath(gt_plot_path, start=html_path)
        out_rel_path = os.path.relpath(out_path, start=html_path)
        # save html to file
        html_str = generate_html(subset, img_task_ids, pred_eval_dict, refvg_loader,
                                 gt_rel_path, out_rel_path)
        html_name = '%s_%s(%s).html' % (subset, len(img_task_ids), total_num)
        with open(os.path.join(html_path, html_name), 'w') as f:
            f.write(html_str)
        if verbose:
            print('%s saved.' % html_name)
    return


def generate_html(subset, img_task_ids, pred_dict, refvg_loader, gt_rel_path, out_rel_path):
    title = '%s:%d' % (subset, len(img_task_ids))
    html_str = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>%s</title>
</head>
<body>
<h1>Visualization of Predictions</h1>
<h3>Predictions: %s</h3>
<h3>Subset: %s </h3>
<h3>Visualization count: %d</h3>
<hr>
''' % (title, '; '.join(pred_dict.keys()), subset, len(img_task_ids))

    if result_rel_path is not None:
        html_str += '<h3>Results</h3><object data="' + result_rel_path + '" width="800" height="800">' \
                                                                         'TXT Object Not supported</object><hr>\n'

    for img_id, task_id in img_task_ids:
        img_data = refvg_loader.get_img_ref_data(img_id)
        task_i = img_data['task_ids'].index(task_id)
        t_name = '%s_%s' % (img_id, task_id)

        phrase = img_data['phrases'][task_i]
        p_structure = img_data['p_structures'][task_i]
        c_a_r = '|'.join(p_structure['attributes']) + ' || ' + p_structure['name'] + ' || ' \
                + '|'.join(r['predicate'] for r in p_structure['relations'])
        task_pred = pred[img_id][task_id]
        iou_box = task_pred['iou_box']
        iou_mask = task_pred['iou_mask']
        subsets = task_pred['subsets']

        html_str += '<h3>[%s]: %s (%s)</h3>\n' % (task_id, phrase, c_a_r)
        html_str += '<h3>box_iou: %.4f; mask_iou: %.4f</h3>\n' % (iou_box, iou_mask)
        html_str += '<h3>Subsets: %s </h3>\n' % ', '.join(subsets)
        gt_f = os.path.join(gt_rel_path, '%s.jpg' % t_name)
        pred_f = os.path.join(pred_rel_path, '%s.jpg' % t_name)
        html_str += '<img src="%s" style="width:500px;">\n' % gt_f
        html_str += '<img src="%s" style="width:500px;">\n<hr>\n\n' % pred_f
    html_str += '</body>\n</html>\n'
    return html_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_path', type=str, required=True,
                        help='path to the save prediction file, after evaluation.')
    parser.add_argument('-o', '--out_path', type=str, default=None, help='path to save output html files')
    parser.add_argument('-g', '--gt_plot_path', type=str, default='data/refvg/visualizations',
                        help='path to saved gt plots')
    parser.add_argument('-v', '--pred_plot_path', type=str, default=None, help='path to saved pred plots')
    parser.add_argument('-r', '--result_path', type=str, default=None, help='path to saved result txt file')
    parser.add_argument('-s', '--split', type=str, default='miniv',
                        help='dataset split to visualize: val, miniv, test, train, val_miniv, etc')
    parser.add_argument('-n', '--all_task_num', type=int, default=20,
                        help='Maximum number of tasks to visualize for "all"')
    parser.add_argument('-m', '--sub_task_num', type=int, default=10,
                        help='Maximum number of tasks to visualize for each subset')
    args = parser.parse_args()

    parent_dir = os.path.dirname(args.pred_path)
    if args.out_path is None:
        args.out_path = os.path.join(parent_dir, 'html/')

    exp_name = os.path.basename(args.pred_path)[10:-4]  # 'pred-eval_%s.npy' % exp_name
    if args.pred_plot_path is None:
        pred_name = '-'.join(exp_name.split('-')[:-1])
        args.pred_plot_path = os.path.join(parent_dir, 'pred_plots_%s/' % pred_name)
    if args.result_path is None:
        args.result_path = os.path.join(parent_dir, 'results_%s.txt' % exp_name)

    visualize(None, 'vg_gt', pred_eval_path=args.pred_path, exp_path=args.out_path, gt_plot_path=args.gt_plot_path,
              refvg_split=args.split, all_task_num=args.all_task_num, subset_task_num=args.sub_task_num, verbose=True)
    return


if __name__ == '__main__':
    main()


# Deprecated: See it in PhraseCutEnsemble
# def evaluate_per_cat(predictions, loader=None, pred_name='rand_vg', split='val', subset=True, eval_img_count=0,
#                      out_path='output/eval_refvg/temp'):
