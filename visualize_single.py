from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
current_path = os.path.realpath(__file__)
dataset_path = os.path.join(current_path, '..')
sys.path.append(os.path.abspath(dataset_path))

import utils.subset as subset_utils
from utils.visualize_utils import gt_visualize_to_file, pred_visualize_to_file, score_visualize_to_file
from utils.refvg_loader import RefVGLoader


def visualize(pred_eval=None, exp_name='temp', pred_eval_path=None, pred_score_thresh=0.0, out_path=None,
              gt_plot_path='data/refvg/visualizations', refvg_loader=None, refvg_split=None, all_task_num=400,
              subset_task_num=200, verbose=True, gt_skip_exist=True, pred_skip_exist=True):
    # prepare
    if pred_eval is not None:
        predictions = pred_eval
    else:
        predictions = np.load(pred_eval_path).item()
        exp_name = pred_eval_path.split('/')[-2]  # 'out_path/{exp_name}/pred_eval.npy'
    assert isinstance(predictions, dict)
    if refvg_loader is None:
        refvg_loader = RefVGLoader(split=refvg_split)

    html_path = os.path.join(out_path, 'htmls')
    pred_plot_path = os.path.join(out_path, 'pred_plots')
    if not os.path.exists(html_path):
        os.makedirs(html_path)
    if not os.path.exists(pred_plot_path):
        os.makedirs(pred_plot_path)
    if not os.path.exists(gt_plot_path):
        os.makedirs(gt_plot_path)

    result_path = os.path.join(out_path, 'results.txt')
    result_enabled = os.path.exists(result_path)

    score_plot_enabled = 'pred_scores' in list(list(predictions.values())[0].values())[0]
    score_plot_path = None
    if score_plot_enabled:
        score_plot_path = os.path.join(out_path, 'score_plots')
        if not os.path.exists(score_plot_path):
            os.makedirs(score_plot_path)

    if subset_task_num > 0:
        subsets = subset_utils.subsets
    else:
        subsets = ['all']
    subset_dict = dict()
    for s in subsets:
        subset_dict[s] = list()
    for img_id, img_pred in predictions.items():
        for task_id, pred in img_pred.items():
            if 'subsets' in pred:
                for subset in pred['subsets']:
                    subset_dict[subset].append((img_id, task_id))
            else:
                pred['subsets'] = ['all']
                subset_dict['all'].append((img_id, task_id))

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
            pred = predictions[img_id][task_id]
            if 'pred_mask' not in pred:
                assert 'pred_scores' in pred
                pred_mask = pred['pred_scores'] >= pred_score_thresh
            else:
                pred_mask = pred['pred_mask']
                pred_mask = np.unpackbits(pred_mask)[:img_data['height'] * img_data['width']]\
                    .reshape((img_data['height'], img_data['width']))

            g1 = gt_visualize_to_file(img_data, task_id, out_path=gt_plot_path, skip_exist=gt_skip_exist)
            g2 = pred_visualize_to_file(img_data, task_id, pred_mask=pred_mask, out_path=pred_plot_path,
                                        skip_exist=pred_skip_exist)

            g3 = False
            if 'pred_scores' in pred:
                score_mask = pred['pred_scores']
                g3 = score_visualize_to_file(img_data, task_id, score_mask=score_mask, out_path=score_plot_path,
                                             skip_exist=pred_skip_exist)
            if verbose:
                print('img %d exp %s: gt plot - %s, pred plot - %s, score plot - %s' % (i, exp_name, g1, g2, g3))

        # html parameters
        gt_rel_path = os.path.relpath(os.path.abspath(gt_plot_path), start=html_path)
        pred_rel_path = os.path.relpath(os.path.abspath(pred_plot_path), start=html_path)
        result_rel_path = None
        if result_enabled:
            result_rel_path = os.path.relpath(os.path.abspath(result_path), start=html_path)
        score_rel_path = None
        if score_plot_enabled:
            score_rel_path = os.path.relpath(os.path.abspath(score_plot_path), start=html_path)
        # save html to file
        html_str = generate_html(exp_name, subset, img_task_ids, predictions, refvg_loader,
                                 gt_rel_path, pred_rel_path, score_rel_path, result_rel_path)
        html_name = '%s_%s(%s).html' % (subset, len(img_task_ids), total_num)
        with open(os.path.join(html_path, html_name), 'w') as f:
            f.write(html_str)
        if verbose:
            print('%s saved.' % html_name)
    return


def generate_html(exp_name, subset, img_task_ids, pred, refvg_loader, gt_rel_path, pred_rel_path, score_rel_path,
                  result_rel_path):
    title = '%s:%s:%d' % (subset, exp_name, len(img_task_ids))
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
<h3>Left: ground-truth; Right: prediction</h3>
<hr>
''' % (title, exp_name, subset, len(img_task_ids))

    if result_rel_path is not None:
        html_str += '<h3>Results</h3><object data="' + result_rel_path + '" width="800" height="800">' \
                                                                         'TXT Object Not supported</object><hr>\n'

    for img_id, task_id in img_task_ids:
        img_data = refvg_loader.get_img_ref_data(img_id)
        task_i = img_data['task_ids'].index(task_id)
        t_name = '%s.jpg' % task_id

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
        if 'tag' in task_pred:
            html_str += '<h3>Tag: %s </h3>\n' % task_pred['tag']

        gt_f = os.path.join(gt_rel_path, t_name)
        pred_f = os.path.join(pred_rel_path, t_name)
        html_str += '<img src="%s" style="width:400px;">\n' % gt_f
        html_str += '<img src="%s" style="width:400px;">\n' % pred_f
        if score_rel_path is not None:
            score_f = os.path.join(score_rel_path, t_name)
            html_str += '<img src="%s" style="width:420px;">\n' % score_f
        html_str += '<hr>\n'

    html_str += '</body>\n</html>\n'
    return html_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_path', type=str, required=True,
                        help='path to the save prediction file, after evaluation.')
    parser.add_argument('-o', '--out_path', type=str, default=None, help='path to save output files')
    parser.add_argument('-g', '--gt_plot_path', type=str, default='data/refvg/visualizations',
                        help='path to saved gt plots')
    parser.add_argument('-s', '--split', type=str, default='miniv',
                        help='dataset split to visualize: val, miniv, test, train, val_miniv, etc')
    parser.add_argument('-n', '--all_task_num', type=int, default=400,
                        help='Maximum number of tasks to visualize for "all"')
    parser.add_argument('-m', '--sub_task_num', type=int, default=200,
                        help='Maximum number of tasks to visualize for each subset')
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = os.path.dirname(args.pred_path)

    visualize(None, 'imp_eval', pred_eval_path=args.pred_path, out_path=args.out_path, gt_plot_path=args.gt_plot_path,
              refvg_split=args.split, all_task_num=args.all_task_num, subset_task_num=args.sub_task_num, verbose=True)
    return


if __name__ == '__main__':
    main()


# Deprecated: See it in PhraseCutEnsemble
# def evaluate_per_cat(predictions, loader=None, pred_name='rand_vg', split='val', subset=True, eval_img_count=0,
#                      out_path='output/eval_refvg/temp'):
