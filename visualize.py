import argparse
import random
import os
import numpy as np

import utils.subset as subset_utils
from utils.visualizer import Visualizer


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
