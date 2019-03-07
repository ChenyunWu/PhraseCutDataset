from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.switch_backend('agg')
current_path = os.path.realpath(__file__)
dataset_path = os.path.join(current_path, '..')
sys.path.append(os.path.abspath(dataset_path))

from utils.iou import *
from utils.subset import *
from utils.eval_utils import *


def evaluate(predictions, pred_name='rand_vg', split='val', subset=True,
             eval_img_count=-1, visualize_img_count=0):

    # initialize
    loader = RefVGLoader(split=split)
    loader.shuffle()
    if eval_img_count < 0:
        eval_img_count = len(loader.ref_img_ids)

    # stats for each subset: correct_count, iou_pred_box, iou_pred_mask
    stats = {'all': [0, [], []]}
    if subset:
        for k in subsets:
            stats[k] = [0, [], []]

    out_path = 'output/eval_refvg/%s' % pred_name
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plots = {}
    to_plot = 0
    pdf = None
    if visualize_img_count > 0:
        pdf = PdfPages(os.path.join(out_path, 'visualizations_%s.pdf' % pred_name))
        to_plot = len(stats.keys())
        for k in stats.keys():
            fig, axes = plt.subplots(int((visualize_img_count + 2) / 3), 3, figsize=(8.5, 1.1 * visualize_img_count))
            plots[k] = [fig, axes, 0]

    correct_count = 0
    task_count = 0
    for img_i, img_id in enumerate(predictions.keys()):

        img_data = loader.get_img_ref_data(img_id)
        for task_i, task_id in enumerate(img_data['task_ids']):
            # predictions
            if task_id not in predictions[img_id]:
                continue
            task_count += 1
            pred = predictions[img_id][task_id]
            correct = pred.get('correct', 0)
            pred_boxes = pred.get('pred_boxes', [[0, 0, 1, 1]])
            pred_mask_bin = pred.get('pred_mask', None)
            if pred_mask_bin is None:
                pred_mask = np.zeros((img_data['height'], img_data['width']))
            else:
                pred_mask = np.unpackbits(pred_mask_bin)[:img_data['height'] * img_data['width']]\
                    .reshape((img_data['height'], img_data['width']))
            can_boxes = pred.get('can_boxes', None)

            # ground-truth
            gt_boxes = img_data['gt_boxes'][task_i]
            gt_Polygons = img_data['gt_Polygons'][task_i]
            gt_polygons = []
            for ps in gt_Polygons:
                gt_polygons += ps

            # compute stats
            correct_count += correct
            iou_pred_box = iou_boxes(pred_boxes, gt_boxes)
            iou_pred_mask = iou_polygons_masks(gt_polygons, [pred_mask])

            # log subset stats, make visualizations
            phrase = img_data['phrases'][task_i]
            phrase_structure = img_data['p_structures'][task_i]
            cond = {'all': True}
            if subset:
                cond = get_subset(phrase, phrase_structure, gt_boxes)
            visualized = False
            for k, v in cond.items():
                if v:
                    stats[k][0] += correct
                    stats[k][1].append(float(iou_pred_box))
                    stats[k][2].append(float(iou_pred_mask))

                    # visualize
                    if not visualized and visualize_img_count > 0:
                        fig, axes, v_count = plots[k]
                        if -1 < v_count < visualize_img_count:
                            visualized = True
                            ax = axes.flatten()[v_count]
                            visualize(ax, img_data, task_i, pred_boxes, pred_mask, can_boxes, iou_pred_box,
                                      iou_pred_mask)
                            plots[k][2] += 1
                            print('visualised %s-%d:%s' % (k, v_count, task_id))
                            if plots[k][2] == visualize_img_count:
                                fig.suptitle('Subset: ' + k)
                                fig.tight_layout(rect=[0, 0, 1, 0.97])
                                pdf.savefig(fig, dpi=300)
                                plt.close(fig)
                                print('%s visualization done.' % k)
                                to_plot -= 1
                                plots[k][2] = -1
                                if to_plot == 0:
                                    pdf.close()

        # print
        print('image[%d/%d] %d phrases. Up till now: box_acc %.3f, mean_iou_box %.3f, mean_iou_mask %.3f' %
              (img_i, eval_img_count, len(img_data['task_ids']), stats['all'][0] * 1.0 / task_count,
               np.mean(stats['all'][1]), np.mean(stats['all'][2])))

        # if we've done enough images
        if img_i + 1 == eval_img_count:
            break

    if to_plot > 0:
        for k in plots.keys():
            fig, _, v_count = plots[k]
            if v_count >= 0:
                fig.suptitle('Subset: ' + k)
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig, dpi=300)
                plt.close(fig)
                print('%s visualization %d.' % (k, v_count))
        pdf.close()
    print('Start to analyze %s:' % pred_name)
    results = analyze_subset_stats(stats, out_path, pred_name, eval_img_count)
    return results, stats


def analyze_subset_stats(stats, out_path, pred_name, img_num):
    subset_results = {}
    result_f = open(os.path.join(out_path, 'results_%d.txt' % img_num), 'w')
    summary_mask = open('output/eval_refvg/summary_mask.csv', 'a')
    summary_box = open('output/eval_refvg/summary_box.csv', 'a')
    summary_subset = open('output/eval_refvg/summary_subset.csv', 'a')
    subset_summary_str = '%s,%d' % (pred_name, img_num)

    for k, v in sorted(stats.items()):
        count = len(v[1])
        if count == 0:
            print('\n%s: count = 0')
            continue
        box_acc = v[0] * 1.0 / count
        mean_pred_box = np.mean(v[1])
        mean_pred_mask = np.mean(v[2])
        str = '\n%s: count=%d, box_acc=%.3f, mean_iou: pred_box=%.3f, pred_mask=%.3f' \
              % (k, count, box_acc, mean_pred_box, mean_pred_mask)
        print(str)
        result_f.write(str + '\n')
        subset_summary_str += ',%.3f' % mean_pred_mask

        box_threshs = [0.1, 0.3, 0.5, 0.7, 0.9]
        mask_threshs = [0.1, 0.3, 0.5, 0.7, 0.9]

        pred_box_acc = {}
        pred_box_acc_str = 'pred_box_acc: '
        box_sum_str = '%s,%d,%.3f' % (pred_name, img_num, mean_pred_box)
        for thresh in box_threshs:
            pred_box_acc[thresh] = np.sum(np.array(v[1]) > thresh) * 1.0 / count
            pred_box_acc_str += 'acc%.1f = %.3f  ' % (thresh, pred_box_acc[thresh])
            box_sum_str += ',%.3f' % pred_box_acc[thresh]
        print(pred_box_acc_str)
        result_f.write(pred_box_acc_str + '\n')
        if k == 'all':
            summary_box.write(box_sum_str + '\n')

        pred_mask_acc = {}
        pred_mask_acc_str = 'pred_mask_acc: '
        mask_sum_str = '%s,%d,%.3f' % (pred_name, img_num, mean_pred_mask)
        for thresh in mask_threshs:
            pred_mask_acc[thresh] = np.sum(np.array(v[2]) > thresh) * 1.0 / count
            pred_mask_acc_str += 'acc%.1f = %.3f  ' % (thresh, pred_mask_acc[thresh])
            mask_sum_str += ',%.3f' % pred_mask_acc[thresh]
        print(pred_mask_acc_str)
        result_f.write(pred_mask_acc_str)
        if k == 'all':
            summary_mask.write(mask_sum_str + '\n')

        result = {'pred_box_acc': pred_box_acc, 'mean_pred_box': mean_pred_box,
                  'pred_mask_acc': pred_mask_acc, 'mean_pred_mask': mean_pred_mask,
                  'box_acc': box_acc}
        subset_results[k] = result

    result_f.close()
    summary_mask.close()
    summary_box.close()
    summary_subset.write(subset_summary_str + '\n')
    summary_subset.close()
    return subset_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_name', type=str, help='name of the predictor to be evaluated: vg_pred, rand_vg_pred',
                        default='vg_gt')
    parser.add_argument('--pred_path', type=str, help='path to the prediction results. If None, run the predictor.',
                        default=None)  #'output/eval_refvg/vg_pred/val_50.npy')
    parser.add_argument('--split', type=str, help='dataset split to evaluate: val, miniv, test, train, val_miniv, etc',
                        default='val')
    parser.add_argument('--eval_img_count', type=int, help='number of images to evaluate. -1 means the whole split',
                        default=500)
    parser.add_argument('--visualize_img_count', type=int, help='number of images to visualize per subset',
                        default=12)
    parser.add_argument('--subset', type=int, default=1, help='whether to enable subset analysis')
    parser.add_argument('--save_pred', type=int, default=1, help='whether to save predictions to file.')

    # parameters for det experiments: score_thresh=0.1, max_can=10, sort_label='gt'
    parser.add_argument('--det_score_thresh', type=float, default=0.1, help='score threshold for detected candidates')
    parser.add_argument('--det_max_can', type=int, default=10, help='max number of detected candidates')
    parser.add_argument('--det_sort_label', type=str, default='gt', help='how to rank the candidates. '
                         '"gt": logits of the phrase name category; "det": scores of the detected category')

    args = parser.parse_args()

    # make other options
    out_path = None
    if args.save_pred:
        out_path = 'output/eval_refvg/' + args.pred_name
        if args.pred_name.startswith('det'):
            out_path += '_%.1f_%d_%s' % (args.det_score_thresh, args.det_max_can, args.det_sort_label)
    if not args.pred_path or not os.path.exists(args.pred_path):
        if args.pred_name == 'vg_gt':
                predictions = vg_gt_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path)
        elif args.pred_name == 'vg_rand':
            predictions = vg_rand_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path)

        elif args.pred_name == 'ins_rand':
            predictions = ins_rand_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path)
        elif args.pred_name == 'ins_mattnet_pred':
            from _comprehend.eval.mattnet_ins_predictor import mattnet_ins_predictor
            predictions = mattnet_ins_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path)

        elif args.pred_name.startswith('det_rand'):
            from _comprehend.eval.det_baseline_predictor import det_rand
            predictions = det_rand(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path,
                                   score_thresh=args.det_score_thresh, max_can=args.det_max_can,
                                   sort_label=args.det_sort_label)
        elif args.pred_name.startswith('det_upperbound'):
            from _comprehend.eval.det_baseline_predictor import det_upperbound
            predictions = det_upperbound(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path,
                                         score_thresh=args.det_score_thresh, max_can=args.det_max_can,
                                         sort_label=args.det_sort_label)
        elif args.pred_name.startswith('det_mattnet_pred'):
            from _comprehend.eval.mattnet_det_predictor import mattnet_det_predictor
            predictions = mattnet_det_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path,
                                                score_thresh=args.det_score_thresh, max_can=args.det_max_can,
                                                sort_label=args.det_sort_label)
        elif args.pred_name.startswith('rmi_pred'):
            from _rmi.rmi_refvg_predictor import rmi_refvg_predictor
            predictions = rmi_refvg_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path)
        else:
            raise NotImplementedError
    else:
        predictions = np.load(args.pred_path).item()
    evaluate(predictions=predictions, pred_name=args.pred_name, split=args.split, subset=args.subset,
             eval_img_count=args.eval_img_count, visualize_img_count=args.visualize_img_count)


