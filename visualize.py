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

html_head_str_formatter = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>%s</title>
</head>
<body>
<h1>Visualization of Predictions</h1>
<h3>Subset: %s </h3>
<h3>Visualization count: %d</h3>
<h3>%s</h3>
<hr>
'''

html_fig_str_formatter = '''
<figure style="display:inline-block">
  <img src="%s" style="width:300px;">
  <figcaption>%s</figcaption>
</figure>
'''


class Visualizer:
    def __init__(self, refvg_loader=None, refvg_split=None, pred_plot_path=None,
                 gt_plot_path='data/refvg/visualizations', pred_skip_exist=True, gt_skip_exist=True,
                 all_task_num=400, subset_task_num=200, include_subsets=None):

        if refvg_loader is None:
            refvg_loader = RefVGLoader(split=refvg_split)
        self.refvg_loader = refvg_loader

        self.pred_skip_exist = pred_skip_exist
        self.gt_skip_exist = gt_skip_exist
        self.all_task_num = all_task_num
        self.subset_task_num = subset_task_num

        self.gt_plot_path = gt_plot_path
        self.pred_bin_path = os.path.join(pred_plot_path, 'pred_bin')
        self.pred_score_path = os.path.join(pred_plot_path, 'pred_score')
        self.pred_box_path = os.path.join(pred_plot_path, 'pred_box')

        if not os.path.exists(gt_plot_path):
            os.makedirs(gt_plot_path)

        self.tasks_plotted_cache = dict()
        self.tasks_in_subset = dict()

        self.include_subsets = include_subsets
        if include_subsets is None:
            self.include_subsets = subset_utils.subsets
        if subset_task_num <= 0:
            self.include_subsets = ['all']
        if 'all' not in self.include_subsets:
            self.include_subsets.insert(0, 'all')
        for subset in self.include_subsets:
            self.tasks_in_subset[subset] = set()

        self.tasks_html_str = dict()

    def is_enough_plots(self, all_task_num=-1, subset_task_num=-1):
        if all_task_num < 0:
            all_task_num = self.all_task_num
        if subset_task_num < 0:
            subset_task_num = self.subset_task_num
        if len(self.tasks_in_subset['all']) < all_task_num:
            return False
        for s in self.tasks_in_subset.values():
            if len(s) < subset_task_num:
                return False
        return True

    def task_is_needed(self, img_id, task_id):
        if len(self.tasks_in_subset['all']) < self.all_task_num:
            return True
        task_subsets = self.refvg_loader.get_task_subset(img_id, task_id)
        for sub in task_subsets:
            if sub in self.tasks_in_subset:
                if len(self.tasks_in_subset[sub]) < self.subset_task_num:
                    return True
        return False

    def plot_single_task(self, img_id, task_id, task_pred_dict, pred_bin_tags=None, pred_score_tags=None,
                         pred_box_tags=None, verbose=False, range01=True):
        fig_name = '%s.jpg' % task_id
        img_data = self.refvg_loader.get_img_ref_data(img_id)
        task_subsets = self.refvg_loader.get_task_subset(img_id, task_id)
        for subset in task_subsets:
            if subset in self.tasks_in_subset:
                self.tasks_in_subset[subset].add(task_id)

        if task_id not in self.tasks_plotted_cache:
            self.tasks_plotted_cache[task_id] = dict()
        task_cache_dict = self.tasks_plotted_cache[task_id]
        task_cache_dict['header'] = self._gen_task_html_header(img_data, task_id, task_subsets, task_pred_dict)

        fig_path = os.path.join(self.gt_plot_path, fig_name)
        is_new_plot = gt_visualize_to_file(img_data, task_id, fig_path=fig_path, skip_exist=self.gt_skip_exist)
        plot_info = 'task(%d) %s: plot gt:%s;' % (len(self.tasks_plotted_cache) + 1, task_id, is_new_plot)
        tag = 'Ground Truth'
        if not is_new_plot:
            tag += '(old plot)'
        task_cache_dict['figs'] = [(tag, fig_path)]

        if pred_bin_tags is not None:
            for tag in pred_bin_tags:
                pred_bin = task_pred_dict[tag]
                if len(pred_bin.shape) == 1:
                    pred_bin = np.unpackbits(pred_bin)[:img_data['height'] * img_data['width']] \
                        .reshape((img_data['height'], img_data['width']))
                out_path = os.path.join(self.pred_bin_path, tag)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                fig_path = os.path.join(out_path, fig_name)
                is_new_plot = pred_visualize_to_file(img_data, fig_path=fig_path, pred_mask=pred_bin,
                                                     skip_exist=self.pred_skip_exist)
                plot_info += 'bin-%s:%s;' % (tag, is_new_plot)
                if tag + '_info' in task_pred_dict:
                    tag += ': ' + task_pred_dict[tag + '_info']
                if not is_new_plot:
                    tag += ' (old plot)'
                task_cache_dict['figs'].append((tag, fig_path))

        if pred_box_tags is not None:
            for tag in pred_box_tags:
                pred_boxlist = task_pred_dict[tag]
                out_path = os.path.join(self.pred_box_path, tag)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                fig_path = os.path.join(out_path, fig_name)
                is_new_plot = pred_visualize_to_file(img_data, fig_path=fig_path, pred_boxlist=pred_boxlist,
                                                     skip_exist=self.pred_skip_exist)
                plot_info += 'box-%s:%s;' % (tag, is_new_plot)
                if tag + '_info' in task_pred_dict:
                    tag += ': ' + task_pred_dict[tag + '_info']
                if not is_new_plot:
                    tag += ' (old plot)'
                task_cache_dict['figs'].append((tag, fig_path))

        if pred_score_tags is not None:
            task_cache_dict['figs2'] = list()
            for tag in pred_score_tags:
                pred_score = task_pred_dict[tag]
                out_path = os.path.join(self.pred_score_path, tag)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                fig_path = os.path.join(out_path, fig_name)
                cb = True
                if range01:
                    cb = tag == 'pred_scores'
                is_new_plot = score_visualize_to_file(img_data, fig_path=fig_path, score_mask=pred_score,
                                                      skip_exist=self.pred_skip_exist,
                                                      include_cbar=cb, range01=range01)
                plot_info += 'score-%s:%s;' % (tag, is_new_plot)
                if tag + '_info' in task_pred_dict:
                    tag += ': ' + task_pred_dict[tag + '_info']
                if not is_new_plot:
                    tag += ' (old plot)'
                task_cache_dict['figs2'].append((tag, fig_path))

        if verbose:
            print(plot_info)
        return

    @staticmethod
    def _gen_task_html_header(img_data, task_id, task_subsets, task_pred_dict):
        task_i = img_data['task_ids'].index(task_id)
        phrase = img_data['phrases'][task_i]
        p_structure = img_data['p_structures'][task_i]
        c_a_r = ' | '.join(p_structure['attributes']) + ' || ' + p_structure['name'] + ' || ' \
                + ' | '.join(['%s (%s)' % (pn[0], pn[1]) for pn in p_structure['relation_descriptions']])
        task_header = '<h3>[%s]: %s (%s)</h3>\n' % (task_id, phrase, c_a_r)
        iou_box = task_pred_dict.get('iou_box', -1)
        iou_mask = task_pred_dict.get('iou_mask', -1)
        if iou_mask > 0 or iou_box > 0:
            task_header += '<h3>box_iou: %.4f; mask_iou: %.4f</h3>\n' % (iou_box, iou_mask)
        task_header += '<h3>Subsets: %s </h3>\n' % ', '.join(task_subsets)
        if 'info' in task_pred_dict:
            task_header += '<h3>Info: %s </h3>\n' % task_pred_dict['info']
        return task_header

    def _single_task_html_str(self, task_id, html_path):
        # html parameters
        task_cache = self.tasks_plotted_cache[task_id]
        html_str = task_cache['header']

        for fig_tag, fig_path in task_cache['figs']:
            rel_path = os.path.relpath(os.path.abspath(fig_path), start=html_path)
            html_str += html_fig_str_formatter % (rel_path, fig_tag)
        if len(task_cache['figs']) > 3 and 'figs2' in task_cache:
            html_str += '<br>\n'
        for fig_tag, fig_path in task_cache.get('figs2', list()):
            rel_path = os.path.relpath(os.path.abspath(fig_path), start=html_path)
            html_str += html_fig_str_formatter % (rel_path, fig_tag)

        html_str += '<hr>\n\n'
        return html_str

    def generate_html(self, html_path, enable_subsets=True, result_txt_path=None, extra_info=''):
        if enable_subsets:
            subsets = self.tasks_in_subset.keys()
        else:
            subsets = ['all']

        if not os.path.exists(html_path):
            os.makedirs(html_path)

        if html_path not in self.tasks_html_str:
            self.tasks_html_str[html_path] = dict()
        tasks_html_str_dict = self.tasks_html_str[html_path]

        for subset in subsets:
            html_str = html_head_str_formatter % (subset, subset, len(self.tasks_in_subset[subset]), extra_info)

            if result_txt_path is not None:
                result_rel_path = os.path.relpath(os.path.abspath(result_txt_path), start=html_path)
                html_str += '<h3>Results</h3><object data="' + result_rel_path \
                            + '" width="1200" height="800">TXT Object Not supported</object><hr>\n'

            for task_id in self.tasks_in_subset[subset]:
                if task_id not in tasks_html_str_dict:
                    tasks_html_str_dict[task_id] = self._single_task_html_str(task_id, html_path)
                html_str += tasks_html_str_dict[task_id] + '\n'

            html_str += '</body>\n</html>\n'

            html_name = '%s.html' % subset
            with open(os.path.join(html_path, html_name), 'w') as f:
                f.write(html_str)
            print('%s saved to %s.' % (html_name, html_path))
        return


def visualize_from_pred_path(pred_eval_path=None, refvg_split=None, out_path=None,
                             gt_plot_path='data/refvg/visualizations', all_task_num=400, subset_task_num=200,
                             gt_skip_exist=True, pred_skip_exist=True, verbose=True):

    predictions = np.load(pred_eval_path).item()
    assert isinstance(predictions, dict)

    # sample
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
                    if subset in subset_dict:
                        subset_dict[subset].append((img_id, task_id))
            else:
                pred['subsets'] = ['all']
                subset_dict['all'].append((img_id, task_id))

    to_plot = list()
    for subset in subset_utils.subsets:
        if subset not in subset_dict:
            continue
        img_task_ids = subset_dict[subset]
        if subset == 'all':
            sample_num = all_task_num
        else:
            sample_num = subset_task_num
        if len(img_task_ids) > sample_num:
            img_task_ids = random.sample(img_task_ids, sample_num)
        to_plot += img_task_ids

    # plot
    visualizer = Visualizer(refvg_split=refvg_split, pred_plot_path=os.path.join(out_path, 'pred_plots'),
                            gt_plot_path=gt_plot_path, pred_skip_exist=pred_skip_exist, gt_skip_exist=gt_skip_exist)
    for img_id, task_id in to_plot:
        visualizer.plot_single_task(img_id, task_id, predictions[img_id][task_id], pred_bin_tags='pred_mask',
                                    pred_score_tags='pred_scores', verbose=verbose)

    # generate html
    html_path = os.path.join(out_path, 'htmls')
    result_path = os.path.join(out_path, 'results.txt')
    if not os.path.exists(result_path):
        result_path = None

    visualizer.generate_html(html_path, enable_subsets=subset_task_num > 0, result_txt_path=result_path)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_path', type=str, required=True,
                        help='path to the save prediction file, after evaluation.')
    parser.add_argument('-o', '--save_result_to_path', type=str, default=None, help='path to save output files')
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

    visualize_from_pred_path(pred_eval_path=args.pred_path, refvg_split=args.split, out_path=args.out_path,
                             gt_plot_path=args.gt_plot_path, all_task_num=args.all_task_num,
                             subset_task_num=args.sub_task_num, verbose=True)
    return


if __name__ == '__main__':
    main()
