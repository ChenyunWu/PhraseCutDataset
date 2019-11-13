import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
current_path = os.path.realpath(__file__)
dataset_path = os.path.join(current_path, '..')
sys.path.append(os.path.abspath(dataset_path))

from utils.iou import iou_boxes, iou_polygons_masks
from utils.refvg_loader import RefVGLoader
from utils.predictor_examples import vg_gt_predictor, vg_rand_predictor, ins_rand_predictor
from utils import subset as subset_utils


def evaluate(predictions, pred_score_thresh=0, refvg_loader=None, refvg_split='miniv', pred_name='temp',
             out_path='output/eval_refvg/', analyze_subset=True, log_to_summary=False, save_pred=True, save_result=True,
             verbose=True, use_existing_result=False):

    # initialize
    if refvg_loader is None:
        refvg_loader = RefVGLoader(split=refvg_split)
        refvg_loader.shuffle()

    # stats for each subset: correct_count, iou_box, iou_mask, i_mask, u_mask
    stats = {'all': [0, [], [], [], []]}
    if analyze_subset:
        for k in subset_utils.subsets:
            stats[k] = [0, [], [], [], []]

    correct_count = 0
    task_count = 0
    for img_i, img_id in enumerate(predictions.keys()):
        if img_id not in refvg_loader.img_ids:
            print('WARNING: IMG %d from predction is not in RefVG %s. Ignored.' % (img_id, refvg_split))
            continue
        img_data = refvg_loader.get_img_ref_data(img_id)
        for task_i, task_id in enumerate(img_data['task_ids']):
            if task_id not in predictions[img_id]:
                print('WARNING: no prediction: %s_%s' % (img_id, task_id))
                continue
            task_count += 1
            pred = predictions[img_id][task_id]
            # load for convenience
            phrase_structure = img_data['p_structures'][task_i]
            gt_boxes = img_data['gt_boxes'][task_i]
            gt_Polygons = img_data['gt_Polygons'][task_i]
            pred_boxes = pred.get('pred_boxlist', None)
            correct = pred.get('correct', 0)

            if 'iou_mask' in pred and use_existing_result:
                # if compared with gt before, directly load from pred dict
                iou_box = pred['iou_box']
                iou_mask = pred['iou_mask']
                i_mask = pred['i_mask']
                u_mask = pred['u_mask']
                gt_relative_size = pred['gt_relative_size']

            else:
                # compare with gt, log to pred dict
                pred_mask_bin = pred.get('pred_mask', None)
                if pred_mask_bin is None:
                    if 'pred_scores' in pred:
                        pred_mask = pred['pred_scores'] > pred_score_thresh
                    else:
                        pred_mask = None
                else:
                    pred_mask = np.unpackbits(pred_mask_bin)[:img_data['height'] * img_data['width']]\
                        .reshape((img_data['height'], img_data['width']))

                gt_polygons = list()
                for ps in gt_Polygons:
                    gt_polygons += ps

                # compute stats
                correct_count += correct
                iou_box, iou_mask, i_mask, u_mask, gt_relative_size = 0.0, 0.0, 0.0, 0.0, 0.0
                if pred_boxes is not None:
                    iou_box = iou_boxes(pred_boxes, gt_boxes)
                if pred_mask is not None:
                    iou_mask, i_mask, u_mask, gt_relative_size = iou_polygons_masks(gt_polygons, [pred_mask],
                                                                                    iandu=True, gt_size=True)
                pred['iou_box'] = iou_box
                pred['iou_mask'] = iou_mask
                pred['i_mask'] = i_mask
                pred['u_mask'] = u_mask
                pred['gt_relative_size'] = gt_relative_size

            subsets = ['all']
            if analyze_subset:
                if 'subsets' in pred:
                    subsets = pred['subsets']
                else:
                    cond = subset_utils.get_subset(phrase_structure, gt_boxes, gt_relative_size)
                    subsets = [k for k, v in cond.items() if v]
                    pred['subsets'] = subsets

            # analyze (by subsets), make visualizations
            for k in subsets:
                if k not in stats:
                    print(k, subsets, list(stats.keys()))
                stats[k][0] += correct
                stats[k][1].append(float(iou_box))
                stats[k][2].append(float(iou_mask))
                stats[k][3].append(float(i_mask))
                stats[k][4].append(float(u_mask))

        # print
        if verbose:
            print('image[%d/%d] %d phrases. Up till now: box_acc %.3f, mean_iou_box %.3f, mean_iou_mask %.3f' %
                  (img_i, len(predictions), len(img_data['task_ids']), stats['all'][0] * 1.0 / task_count,
                   np.mean(stats['all'][1]), np.mean(stats['all'][2])))

    exp_name = '%s-%s-%d' % (pred_name, refvg_split, len(predictions))
    if (save_pred or save_result) and not os.path.exists(out_path):
        os.makedirs(out_path)
    if save_pred:
        print('saving %s to %s' % (exp_name, out_path))
        pred_path = os.path.join(out_path, 'pred-eval.npy')
        np.save(pred_path, predictions)
    if verbose:
        print('Start to analyze %s:' % exp_name)
    if not save_result:
        out_path = None
    results = analyze_subset_stats(stats, exp_name, out_path, log_to_summary)
    return predictions, results


def analyze_subset_stats(stats, exp_name, out_path, log_to_summary):
    subset_results = {}
    result_f = None
    summary_mask = None
    summary_box = None
    summary_subset = None
    subset_summary_str = exp_name
    if out_path is not None:
        result_f = open(os.path.join(out_path, 'results.txt'), 'w')
    if log_to_summary:
        summary_mask = open('output/eval_refvg/summary_mask.csv', 'a')
        summary_box = open('output/eval_refvg/summary_box.csv', 'a')
        summary_subset = open('output/eval_refvg/summary_subset.csv', 'a')

    sample_num = len(stats['all'][1])
    print('subsets:\n', ','.join([k for k, v in stats.items()]))
    for subset in subset_utils.subsets:
        if subset not in stats:
            subset_summary_str += ',0.0'
            continue
        stat = stats[subset]
        count = len(stat[1])
        if count == 0:
            print('\n%s: count = 0' % subset)
            subset_summary_str += ',0.0'
            continue

        box_acc = stat[0] * 1.0 / count
        mean_box_iou = np.mean(stat[1])
        mean_mask_iou = np.mean(stat[2])
        cum_mask_iou = np.sum(stat[3]) * 1.0 / np.sum(stat[4])
        s = '\n%s: count=%d(%.4f), box_acc=%.4f, mean_box_iou=%.4f, mean_mask_iou=%.4f, cum_mask_iou=%.4f' \
            % (subset, count, count*1.0 / sample_num, box_acc, mean_box_iou, mean_mask_iou, cum_mask_iou)
        print(s)
        if out_path is not None:
            result_f.write(s + '\n')
        if log_to_summary:
            subset_summary_str += ',%.4f' % mean_mask_iou

        box_threshs = [0.5, 0.6, 0.7, 0.8, 0.9]
        pred_box_acc = {}
        pred_box_acc_str = 'pred_box_acc: '
        box_sum_str = '%s,%.4f' % (exp_name, mean_box_iou)

        for thresh in box_threshs:
            pred_box_acc[thresh] = np.sum(np.array(stat[1]) > thresh) * 1.0 / count
            pred_box_acc_str += 'acc%.1f = %.4f  ' % (thresh, pred_box_acc[thresh])
            box_sum_str += ',%.4f' % pred_box_acc[thresh]
        print(pred_box_acc_str)
        if out_path is not None:
            result_f.write(pred_box_acc_str + '\n')
        if log_to_summary and subset == 'all':
            summary_box.write(box_sum_str + '\n')

        mask_threshs = [0.5, 0.6, 0.7, 0.8, 0.9]
        pred_mask_acc = {}
        pred_mask_acc_str = 'pred_mask_acc: '
        mask_sum_str = '%s,%.4f,%.4f' % (exp_name, mean_mask_iou, cum_mask_iou)
        for thresh in mask_threshs:
            pred_mask_acc[thresh] = np.sum(np.array(stat[2]) > thresh) * 1.0 / count
            pred_mask_acc_str += 'acc%.1f = %.4f  ' % (thresh, pred_mask_acc[thresh])
            mask_sum_str += ',%.4f' % pred_mask_acc[thresh]
        print(pred_mask_acc_str)
        if out_path is not None:
            result_f.write(pred_mask_acc_str)
        if log_to_summary and subset == 'all':
            summary_mask.write(mask_sum_str + '\n')

        result = {'pred_box_acc': pred_box_acc, 'mean_box_iou': mean_box_iou,
                  'pred_mask_acc': pred_mask_acc, 'mean_mask_iou': mean_mask_iou, 'cum_mask_iou': cum_mask_iou,
                  'box_acc': box_acc}
        subset_results[subset] = result

    if out_path is not None:
        result_f.close()
    if log_to_summary:
        summary_mask.close()
        summary_box.close()
        summary_subset.write(subset_summary_str + '\n')
        summary_subset.close()
    return subset_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--pred_name', type=str, default='vg_pred', help='name of the predictor to be evaluated:'
                                                                               'vg_gt, vg_rand, ins_rand')
    parser.add_argument('-p', '--pred_path', type=str, default=None,
                        help='path to the prediction results. If empty, run the predictor.')
    parser.add_argument('-s', '--split', type=str, default='miniv',
                        help='dataset split to evaluate: val, miniv, test, train, val_miniv, etc')
    parser.add_argument('-a', '--analyze_subset', type=int, default=1, help='whether to enable subset analysis')
    parser.add_argument('-f', '--save_pred', type=int, default=1, help='whether to save evaluated predictions to file.')
    parser.add_argument('-l', '--log_to_summary', type=int, default=1, help='whether to log results to the summary.')
    args = parser.parse_args()

    out_path = None
    if args.save_pred:
        if args.pred_path is not None:
            out_path = os.path.dirname(args.pred_path)
        else:
            out_path = os.path.join('output/eval_refvg', args.pred_name)

    if not args.pred_path or not os.path.exists(args.pred_path):
        if args.pred_name.starts_with('vg_gt'):
                predictions = vg_gt_predictor(split=args.split, eval_img_count=-1, out_path=out_path)
        elif args.pred_name.starts_with('vg_rand'):
            predictions = vg_rand_predictor(split=args.split, eval_img_count=-1, out_path=out_path)
        elif args.pred_name.starts_with('ins_rand'):
            predictions = ins_rand_predictor(split=args.split, eval_img_count=-1, out_path=out_path)
        else:
            raise NotImplementedError
    else:
        predictions = np.load(args.pred_path).item()

    evaluate(predictions=predictions, refvg_split=args.split, pred_name=args.pred_name, out_path=out_path,
             analyze_subset=args.analyze_subset, log_to_summary=args.log_to_summary, save_pred=args.save_pred)
    return


if __name__ == '__main__':
    main()


# Deprecated: See it in PhraseCutEnsemble
# def evaluate_per_cat(predictions, loader=None, pred_name='rand_vg', split='val', subset=True, eval_img_count=0,
#                      out_path='output/eval_refvg/temp'):
