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


def evaluate(predictions, loader=None, pred_name='rand_vg', split='val', subset=True, eval_img_count=0,
             visualize_img_count=0, out_path='output/eval_refvg/temp', verbose=True):

    # initialize
    if loader is None:
        loader = RefVGLoader(split=split)
        loader.shuffle()
    if eval_img_count < 0:
        eval_img_count = len(loader.img_ids)

    # stats for each subset: correct_count, iou_pred_box, iou_pred_mask, i_pred_mask, u_pred_mask
    stats = {'all': [0, [], [], [], []]}
    if subset:
        for k in subsets:
            stats[k] = [0, [], [], [], []]
    if out_path is not None:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    plots = {}
    to_plot = 0
    pdf = None
    if visualize_img_count > 0:
        pdf = PdfPages(os.path.join(out_path, 'vis_%s_%s%d.pdf' % (pred_name, split, eval_img_count)))
        to_plot = len(stats.keys())
        for k in stats.keys():
            fig, axes = plt.subplots(int((visualize_img_count + 2) / 3), 3, figsize=(8.5, 1.1 * visualize_img_count))
            plots[k] = [fig, axes, 0]

    correct_count = 0
    task_count = 0
    for img_i, img_id in enumerate(predictions.keys()):
        if img_id not in loader.img_ids:
            continue
        img_data = loader.get_img_ref_data(img_id)
        for task_i, task_id in enumerate(img_data['task_ids']):
            # predictions
            if task_id not in predictions[img_id]:
                print('no prediction: %s' % task_id)
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
            iou_pred_mask, i_pred_mask, u_pred_mask, gt_relative_size = iou_polygons_masks(gt_polygons, [pred_mask],
                                                                                           iandu=True, gt_size=True)

            # log subset stats, make visualizations
            phrase = img_data['phrases'][task_i]
            phrase_structure = img_data['p_structures'][task_i]
            cond = {'all': True}
            if subset:
                cond = get_subset(phrase, phrase_structure, gt_boxes, gt_relative_size)
            visualized = False
            for k, v in cond.items():
                if v:
                    stats[k][0] += correct
                    stats[k][1].append(float(iou_pred_box))
                    stats[k][2].append(float(iou_pred_mask))
                    stats[k][3].append(float(i_pred_mask))
                    stats[k][4].append(float(u_pred_mask))

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
        if verbose:
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
    results = analyze_subset_stats(stats, out_path, split, eval_img_count)
    return results, stats


def analyze_subset_stats(stats, out_path, split, img_num):
    subset_results = {}
    result_f = None
    summary_mask = None
    summary_box = None
    summary_subset = None
    exp_str = ''
    subset_summary_str = ''
    if out_path is not None:
        # out_path = 'temp'
        result_f = open(os.path.join(out_path, 'results_%s%d.txt' % (split, img_num)), 'w')
        summary_mask = open('output/eval_refvg/summary_mask.csv', 'a')
        summary_box = open('output/eval_refvg/summary_box.csv', 'a')
        summary_subset = open('output/eval_refvg/summary_subset.csv', 'a')

        exp_str = out_path.split('/')[-1]
        subset_summary_str = exp_str
    sample_num = len(stats['all'][1])
    for k, v in sorted(stats.items()):
        count = len(v[1])
        if count == 0:
            print('\n%s: count = 0' % k)
            continue
        box_acc = v[0] * 1.0 / count
        mean_box_iou = np.mean(v[1])
        mean_mask_iou = np.mean(v[2])
        cum_mask_iou = np.sum(v[3]) * 1.0 / np.sum(v[4])
        str = '\n%s: count=%d(%.4f), box_acc=%.4f, mean_box_iou=%.4f, mean_mask_iou=%.4f, cum_mask_iou=%.4f' \
              % (k, count, count*1.0 / sample_num, box_acc, mean_box_iou, mean_mask_iou, cum_mask_iou)
        print(str)
        if out_path is not None:
            result_f.write(str + '\n')
            subset_summary_str += ',%.4f' % mean_mask_iou

        box_threshs = [0.5, 0.6, 0.7, 0.8, 0.9]
        mask_threshs = [0.5, 0.6, 0.7, 0.8, 0.9]

        pred_box_acc = {}
        pred_box_acc_str = 'pred_box_acc: '
        box_sum_str = '%s,%.4f' % (exp_str, mean_box_iou)
        for thresh in box_threshs:
            pred_box_acc[thresh] = np.sum(np.array(v[1]) > thresh) * 1.0 / count
            pred_box_acc_str += 'acc%.1f = %.4f  ' % (thresh, pred_box_acc[thresh])
            box_sum_str += ',%.4f' % pred_box_acc[thresh]
        print(pred_box_acc_str)
        if out_path is not None:
            result_f.write(pred_box_acc_str + '\n')
            if k == 'all':
                summary_box.write(box_sum_str + '\n')

        pred_mask_acc = {}
        pred_mask_acc_str = 'pred_mask_acc: '
        mask_sum_str = '%s,%.4f,%.4f' % (exp_str, mean_mask_iou, cum_mask_iou)
        for thresh in mask_threshs:
            pred_mask_acc[thresh] = np.sum(np.array(v[2]) > thresh) * 1.0 / count
            pred_mask_acc_str += 'acc%.1f = %.4f  ' % (thresh, pred_mask_acc[thresh])
            mask_sum_str += ',%.4f' % pred_mask_acc[thresh]
        print(pred_mask_acc_str)
        if out_path is not None:
            result_f.write(pred_mask_acc_str)
            if k == 'all':
                summary_mask.write(mask_sum_str + '\n')

        result = {'pred_box_acc': pred_box_acc, 'mean_box_iou': mean_box_iou,
                  'pred_mask_acc': pred_mask_acc, 'mean_mask_iou': mean_mask_iou, 'cum_mask_iou': cum_mask_iou,
                  'box_acc': box_acc}
        subset_results[k] = result
    if out_path is not None:
        result_f.close()
        summary_mask.close()
        summary_box.close()
        summary_subset.write(subset_summary_str + '\n')
        summary_subset.close()
    return subset_results


def evaluate_per_cat(predictions, loader=None, pred_name='rand_vg', split='val', subset=True, eval_img_count=0,
                     out_path='output/eval_refvg/temp'):
    # initialize
    if loader is None:
        loader = RefVGLoader(split=split)
        loader.shuffle()
    if eval_img_count < 0:
        eval_img_count = len(loader.img_ids)

    stats = dict()
    if out_path is not None:
        if not os.path.exists(out_path):
            os.makedirs(out_path)

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
            iou_pred_mask, i_pred_mask, u_pred_mask, gt_relative_size = iou_polygons_masks(gt_polygons, [pred_mask],
                                                                                           iandu=True, gt_size=True)

            # log subset stats, make visualizations
            phrase = img_data['phrases'][task_i]
            phrase_structure = img_data['p_structures'][task_i]
            name = phrase_structure['name']

            if name not in stats:
                # stats for each cat: iou_pred_mask, gt_region_size, att_count, rel_count
                stats[name] = [[], [], 0, 0]

            stats[name][0].append(float(iou_pred_mask))
            stats[name][1].append(float(gt_relative_size))
            if phrase_structure['attributes'] is not None and len(phrase_structure['attributes']) > 0:
                stats[name][2] += 1
            if phrase_structure['relations'] is not None and len(phrase_structure['relations']) > 0:
                stats[name][3] += 1

        print('image[%d/%d] %d phrases.' % (img_i, eval_img_count, len(img_data['task_ids'])))

        # if we've done enough images
        if img_i + 1 == eval_img_count:
            break
    print('Stats:')
    with open(os.path.join(out_path, 'cat_stats.csv'), 'w') as out_f:
        for name, l in stats.items():
            freq = loader.vg_loader.name_to_cnt.get(name, len(l[0]))
            mean_iou = np.mean(np.array(l[0]))
            mean_size = np.mean(np.array(l[1]))
            s = '%s,%d,%.4f,%.4f,%d,%d' % (name, freq, mean_iou, mean_size, l[2], l[3])
            print(s)
            out_f.write(s + '\n')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_name', type=str, help='name of the predictor to be evaluated: vg_pred, rand_vg_pred',
                        default='ensemble_pred')
    parser.add_argument('--pred_path', type=str, help='path to the prediction results. If None, run the predictor.',
                        default='output/eval_refvg/rmi_pred_test0/test.npy')
                        # default='output/eval_refvg/ensemble_IN_obj_cat_msub_wsub_mloc_wloc_mrel_wrel_m_att_logits_topk1.00_test0/test.npy')
    # output/eval_refvg/ensemble_IN_obj_cat_msub_wsub_mloc_wloc_mrel_wrel_m_att_logits_soft0.60_test0/test_2814.npy
    # output/eval_refvg/ensemble_pred_topk1.000000_miniv0/miniv_17.npy
    # 'output/eval_refvg/%s/test_2814.npy'
    # 'ours': 'ensemble_IN_obj_cat_msub_wsub_mloc_wloc_mrel_wrel_m_att_logits_soft0.60_test0',
    # 'MRCN': 'det_upperbound_0.15_1_gt_test0',
    # 'Matt': 'det_mattnet_pred_0.15_50_det_test0',
    # 'RMI': 'rmi_pred_test0'}
    parser.add_argument('--split', type=str, help='dataset split to evaluate: val, miniv, test, train, val_miniv, etc',
                        default='test')
    parser.add_argument('--eval_img_count', type=int, help='number of images to evaluate. <=0 means the whole split',
                        default=0)
    parser.add_argument('--visualize_img_count', type=int, help='number of images to visualize per subset',
                        default=0)
    parser.add_argument('--subset', type=int, default=1, help='whether to enable subset analysis')
    parser.add_argument('--save_pred', type=int, default=1, help='whether to save predictions to file.')

    # parameters for det experiments: score_thresh=0.1, max_can=10, sort_label='gt'
    parser.add_argument('--det_score_thresh', type=float, default=0.1, help='score threshold for detected candidates')
    parser.add_argument('--det_max_can', type=int, default=10, help='max number of detected candidates')
    parser.add_argument('--det_sort_label', type=str, default='gt', help='how to rank the candidates. '
                                     '"gt": logits of the phrase name category; "det": scores of the detected category')

    # parameters for ensemble pred: thresh_by='topk'/'hard'/'soft', thresh
    parser.add_argument('--ensemble_thresh_by', type=str, default='top', help='how to thresh scores: topk/hard/soft')
    parser.add_argument('--ensemble_thresh', type=float, default=1, help='threshold for ensemble scores')
    parser.add_argument('--ensemble_input', type=str, default='obj_cat_msub_wsub_mloc_wloc_mrel_wrel_m_att_scores', help='ensemble input')
    parser.add_argument('--ensemble_exp', type=str, default='', help='ensemble input')

    args = parser.parse_args()

    # make other options
    eval_path = 'output/eval_refvg/' + args.pred_name
    if args.pred_name.startswith('det'):
        eval_path += '_%.2f_%d_%s' % (args.det_score_thresh, args.det_max_can, args.det_sort_label)
    if args.pred_name.startswith('ensemble'):
        if len(args.ensemble_exp) > 0:
            eval_path += '/ensemble_%s_%s%.2f' % (args.ensemble_exp, args.ensemble_thresh_by, args.ensemble_thresh)
        else:
            eval_path += '/ensemble_IN_%s_%s%.2f' % (args.ensemble_input, args.ensemble_thresh_by, args.ensemble_thresh)
    eval_path += '_%s%d' % (args.split, args.eval_img_count)
    out_path = None
    if args.save_pred:
        out_path = eval_path
    if not args.pred_path or not os.path.exists(args.pred_path):
        if args.pred_name == 'vg_gt':
                predictions = vg_gt_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path)
        elif args.pred_name == 'vg_rand':
            predictions = vg_rand_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path)

        elif args.pred_name == 'ins_rand':
            predictions = ins_rand_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path)
        elif args.pred_name == 'ins_mattnet_pred':
            from _comprehend.predictors.mattnet_ins_predictor import mattnet_ins_predictor
            predictions = mattnet_ins_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path)

        elif args.pred_name.startswith('det_rand'):
            from _comprehend.predictors.det_baseline_predictor import det_rand
            predictions = det_rand(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path,
                                   score_thresh=args.det_score_thresh, max_can=args.det_max_can,
                                   sort_label=args.det_sort_label)
        elif args.pred_name.startswith('det_upperbound'):
            from _comprehend.predictors.det_baseline_predictor import det_upperbound
            predictions = det_upperbound(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path,
                                         score_thresh=args.det_score_thresh, max_can=args.det_max_can,
                                         sort_label=args.det_sort_label)
        elif args.pred_name.startswith('det_mattnet_pred'):
            from _comprehend.predictors.mattnet_det_predictor import mattnet_det_predictor
            predictions = mattnet_det_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path,
                                                score_thresh=args.det_score_thresh, max_can=args.det_max_can,
                                                sort_label=args.det_sort_label)
        elif args.pred_name.startswith('rmi_pred'):
            from _rmi.rmi_refvg_predictor import rmi_refvg_predictor
            predictions = rmi_refvg_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path)
        elif args.pred_name.startswith('lstm_pred'):
            from _rmi.lstm_refvg_predictor import lstm_refvg_predictor
            predictions = lstm_refvg_predictor(split=args.split, eval_img_count=args.eval_img_count, out_path=out_path)

        elif args.pred_name.startswith('ensemble_pred'):
            from _ensemble.utils.ensemble_predictor import ensemble_predictor
            if len(args.ensemble_exp) > 0:
                model_path = 'output/ensemble/final/%s' % args.ensemble_exp
            else:
                model_path = 'output/ensemble/final/' \
                             'IN_%s_score_relative_attFalse_quadFalse_GT_box_iou_soft_fc_lr0.010000_bs1024_bn0'\
                             % args.ensemble_input
                             # 'IN_%s_score_relative_attFalse_quadFalse_GT_box_iou_soft_fc128_128_lr0.010000_bs1024_bn0'
                             # 'IN_%s_score_relative_GT_box_iou_soft_fc64_lr0.010000_bs1024_bn0' % args.ensemble_input
            predictions = ensemble_predictor(split=args.split, thresh_by=args.ensemble_thresh_by,
                                             thresh=args.ensemble_thresh, eval_img_count=args.eval_img_count,
                                             out_path=out_path, model_path=model_path, checkpoint='epoch0_step1000.pth')

        else:
            raise NotImplementedError
    else:
        predictions = np.load(args.pred_path).item()

    evaluate(predictions=predictions, pred_name=args.pred_name, split=args.split, subset=args.subset,
             eval_img_count=args.eval_img_count, visualize_img_count=args.visualize_img_count, out_path=eval_path)

    # evaluate_per_cat(predictions=predictions, pred_name=args.pred_name, split=args.split, out_path=eval_path)
