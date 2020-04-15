import os
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

import subset as subset_utils
from visualize_utils import gt_visualize_to_file, pred_visualize_to_file, score_visualize_to_file
from refvg_loader import RefVGLoader
from file_paths import gt_plot_path_gray, gt_plot_path_color, img_fpath


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
    def __init__(self, refvg_loader=None, refvg_split=None, png_path_dict=None, pred_plot_path=None, gt_plot_gray=True,
                 pred_skip_exist=True, gt_skip_exist=True, baselines=None, baselines_skip_exist=True, all_task_num=400,
                 subset_task_num=200, include_subsets=None):

        if refvg_loader is None:
            refvg_loader = RefVGLoader(split=refvg_split)
        self.refvg_loader = refvg_loader

        self.pred_skip_exist = pred_skip_exist
        self.gt_skip_exist = gt_skip_exist
        self.all_task_num = all_task_num
        self.subset_task_num = subset_task_num

        if gt_plot_gray:
            self.gt_plot_path = gt_plot_path_gray
        else:
            self.gt_plot_path = gt_plot_path_color

        if not os.path.exists(self.gt_plot_path):
            os.makedirs(self.gt_plot_path)

        self.pred_bin_path = None
        self.pred_score_path = None
        self.pred_box_path = None
        if pred_plot_path is not None:
            self.pred_bin_path = os.path.join(pred_plot_path, 'pred_bin')
            self.pred_score_path = os.path.join(pred_plot_path, 'pred_score')
            self.pred_box_path = os.path.join(pred_plot_path, 'pred_box')

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

        self.png_path_dict = png_path_dict
        self.baselines = baselines
        self.baselines_skip_exist = baselines_skip_exist
        if baselines is not None:
            for bl_name, bl_dict in baselines.items():
                bl_dict['pred'] = np.load(bl_dict['pred_path'], allow_pickle=True, encoding='latin1').item()
                print('Visualizer: loaded baseline predictions for %s' % bl_name)

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

    def plot_single_task(self, img_id, task_id, task_pred_dict=None,
                         pred_bin_tags=None, pred_score_tags=None, pred_box_tags=None, verbose=False, range01=True):
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

        # raw
        fig_path = os.path.join(img_fpath, '%d.jpg' % img_id)
        task_cache_dict['figs'] = [('raw', fig_path)]

        # gt
        fig_path = os.path.join(self.gt_plot_path, fig_name)
        is_new_plot = gt_visualize_to_file(img_data, task_id, fig_path=fig_path, skip_exist=self.gt_skip_exist)
        plot_info = 'task(%d) %s: plot gt:%s;' % (len(self.tasks_plotted_cache) + 1, task_id, is_new_plot)
        tag = 'Ground Truth'
        if not is_new_plot:
            tag += '(old plot)'
        task_cache_dict['figs'] += [(tag, fig_path)]

        # baselines
        if self.baselines is not None:
            for bl_name, bl_dict in self.baselines.items():
                if img_id not in bl_dict['pred']:
                    print(img_id, 'not in ', bl_name)
                else:
                    predictions = bl_dict['pred']
                    if task_id not in predictions[img_id]:
                        print(task_id, 'not in ', bl_name)
                    else:
                        pred_mask = predictions[img_id][task_id]['pred_mask']
                        pred_mask = np.unpackbits(pred_mask)[:img_data['height'] * img_data['width']]\
                            .reshape((img_data['height'], img_data['width']))
                        fig_path = os.path.join(bl_dict['plot_path'], fig_name)
                        is_new_plot = pred_visualize_to_file(img_data, fig_path=fig_path, pred_mask=pred_mask,
                                                             skip_exist=self.baselines_skip_exist)
                        plot_info += '%s:%s;' % (bl_name, is_new_plot)
                        tag = bl_name
                        if not is_new_plot:
                            tag += '(old plot)'
                        task_cache_dict['figs'].append((tag, fig_path))

        # predictions: use existing png pred paths
        if self.png_path_dict is not None:
            for tag, folder in self.png_path_dict.items():
                png_file_path = os.path.join(folder, '%s.png' % task_id)
                if os.path.exists(png_file_path):
                    task_cache_dict['figs'].append((tag, png_file_path))

        # predictions: make plots
        if pred_bin_tags is not None:
            assert self.pred_bin_path is not None
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
            assert self.pred_box_path is not None
            for tag in pred_box_tags:
                pred_boxlist = task_pred_dict[tag]
                pred_boxes = None
                xywh = True
                if type(pred_boxlist) == list:
                    pred_boxes = pred_boxlist
                    pred_boxlist = None
                    xywh = True
                out_path = os.path.join(self.pred_box_path, tag)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                fig_path = os.path.join(out_path, fig_name)
                is_new_plot = pred_visualize_to_file(img_data, fig_path=fig_path, pred_boxlist=pred_boxlist,
                                                     pred_boxes=pred_boxes, skip_exist=self.pred_skip_exist, xywh=xywh)
                plot_info += 'box-%s:%s;' % (tag, is_new_plot)
                if tag + '_info' in task_pred_dict:
                    tag += ': ' + task_pred_dict[tag + '_info']
                if not is_new_plot:
                    tag += ' (old plot)'
                task_cache_dict['figs'].append((tag, fig_path))

        if pred_score_tags is not None:
            assert self.pred_score_path is not None
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
        iou_box, iou_mask = -1, -1
        if task_pred_dict is not None:
            iou_box = task_pred_dict.get('iou_box', -1)
            iou_mask = task_pred_dict.get('iou_mask', -1)
        if iou_mask > 0 or iou_box > 0:
            task_header += '<h3>box_iou: %.4f; mask_iou: %.4f</h3>\n' % (iou_box, iou_mask)
        task_header += '<h3>Subsets: %s </h3>\n' % ', '.join(task_subsets)
        if task_pred_dict is not None and 'info' in task_pred_dict:
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
