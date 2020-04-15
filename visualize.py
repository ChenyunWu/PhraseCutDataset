import argparse
import random
import os
import numpy as np

import utils.subset as subset_utils
from utils.visualizer import Visualizer
from utils.visualize_utils import png_to_pred_mask


def visualize_from_png_path(refvg_split, png_path, out_path=None, all_task_num=-1, subset_task_num=0,
                            pred_plot=True, gt_skip_exist=True, verbose=True):
    if out_path is None:
        out_path = os.path.normpath(os.path.join(png_path, os.pardir))

    if pred_plot:
        plot_path = os.path.join(out_path, 'pred_plots')
        png_path_dict = None
    else:
        plot_path = None
        png_path_dict = {'prediction': png_path}

    fnames = os.listdir(png_path)
    if all_task_num < 0:
        all_task_num = len(fnames)

    visualizer = Visualizer(refvg_split=refvg_split, png_path_dict=png_path_dict, pred_plot_path=plot_path,
                            pred_skip_exist=True, gt_skip_exist=gt_skip_exist,
                            all_task_num=all_task_num, subset_task_num=subset_task_num)
    random.shuffle(fnames)
    for fname in fnames:
        if not fname.endswith('.png'):
            continue
        task_id = fname.split('.')[0]
        img_id = int(task_id.split('__')[0])
        if visualizer.task_is_needed(img_id, task_id):
            if not pred_plot:
                visualizer.plot_single_task(img_id, task_id, verbose=verbose)
            else:
                pred_mask = png_to_pred_mask(os.path.join(png_path, fname))
                visualizer.plot_single_task(img_id, task_id, {'pred_mask': pred_mask}, pred_bin_tags=('pred_mask',),
                                            verbose=verbose)
            if visualizer.is_enough_plots():
                break

    # generate html
    html_path = os.path.join(out_path, 'htmls')
    result_path = os.path.join(out_path, 'results.txt')
    if not os.path.exists(result_path):
        result_path = None
    visualizer.generate_html(html_path, enable_subsets=subset_task_num > 0, result_txt_path=result_path)
    return


def visualize_from_pred_dict(pred_eval_dict=None, pred_eval_dict_path=None, refvg_split=None, out_path=None,
                             pred_bin_tags=None, pred_score_tags=None, pred_box_tags=None,
                             all_task_num=40, subset_task_num=20, gt_skip_exist=True, pred_skip_exist=True,
                             verbose=True):
    if pred_eval_dict is None:
        predictions = np.load(pred_eval_dict_path, allow_pickle=True).item()
        assert isinstance(predictions, dict)
    else:
        predictions = pred_eval_dict

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
                            pred_skip_exist=pred_skip_exist, gt_skip_exist=gt_skip_exist)
    for img_id, task_id in to_plot:
        visualizer.plot_single_task(img_id, task_id, predictions[img_id][task_id], pred_bin_tags=pred_bin_tags,
                                    pred_score_tags=pred_score_tags, pred_box_tags=pred_box_tags, verbose=verbose)

    # generate html
    html_path = os.path.join(out_path, 'htmls')
    result_path = os.path.join(out_path, 'results.txt')
    if not os.path.exists(result_path):
        result_path = None

    visualizer.generate_html(html_path, enable_subsets=subset_task_num > 0, result_txt_path=result_path)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_path', type=str, default=None,
                        help='path to the folder containing prediction results. If empty, use pred_dict.')
    parser.add_argument('-d', '--pred_dict', type=str, default=None,
                        help='Obsolete. path to the .npy file of prediction results, after evaluation.')
    parser.add_argument('-o', '--out_path', type=str, default=None, help='path to save output files')
    parser.add_argument('-s', '--split', type=str, default='miniv',
                        help='dataset split to visualize: val, miniv, test, train, val_miniv, etc. Must match pred.')
    parser.add_argument('-n', '--all_task_num', type=int, default=40,
                        help='Number of tasks to visualize for "all"')
    parser.add_argument('-m', '--sub_task_num', type=int, default=20,
                        help='Number of tasks to visualize for each subset')
    args = parser.parse_args()
    if args.pred_path is not None:
        if args.out_path is None:
            args.out_path = os.path.normpath(os.path.join(args.pred_path, os.pardir))

        visualize_from_png_path(refvg_split=args.split, png_path=args.pred_path, out_path=args.out_path,
                                all_task_num=args.all_task_num, subset_task_num=args.sub_task_num,
                                gt_skip_exist=True, verbose=True)
    else:
        assert args.pred_dict is not None
        if args.out_path is None:
            args.out_path = os.path.dirname(args.pred_dict)
        visualize_from_pred_dict(pred_eval_dict_path=args.pred_path, refvg_split=args.split, out_path=args.out_path,
                                 pred_bin_tags=['pred_mask'], pred_box_tags=['pred_boxlist'],
                                 all_task_num=args.all_task_num, subset_task_num=args.sub_task_num, verbose=True)
    return


if __name__ == '__main__':
    main()
