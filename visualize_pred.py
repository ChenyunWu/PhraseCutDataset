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
from utils.visualize import visualize_refvg


# colors = {'title': 'black', 'gt_polygons': 'deepskyblue', 'gt_boxes': 'blue', 'gt_all_boxes': 'blue',
#           'vg_boxes': 'green', 'vg_all_boxes': 'green', 'pred_boxes': 'red', 'pred_mask': 'autumn', 'can_boxes': 'red'}

colors = {'title': 'black', 'gt_polygons': 'darkorange', 'gt_boxes': 'chocolate',
          'vg_boxes': 'green', 'vg_all_boxes': 'green', 'pred_boxes': 'deepskyblue', 'pred_mask': 'GnBu',
          'can_boxes': 'darkcyan'}


def do_visualize(pred_dict, can_box_pred, task_ids=[], vis_count=10, split='test', loader=None,
                 out_path='output/visualize/'):

    # initialize
    if loader is None:
        loader = RefVGLoader(split=split)
        loader.shuffle()

    pred_str = '_'.join(pred_dict.keys())
    if can_box_pred is not None:
        pred_str = 'can_' + pred_str
    out_path = os.path.join(out_path, pred_str, split)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    vis_count = max(vis_count, len(task_ids))

    for vis_i in range(vis_count):
        fig, axes = plt.subplots(2, 4, figsize=(8.5, 11))

        # get img_id task_ix, task_id
        if vis_i < len(task_ids):
            task_id = task_ids[vis_i]
            img_id = int(task_id.split('__')[0])
            task_ix = -1
            for task_ix, task in enumerate(loader.ImgReferTasks[img_id]):
                if task['task_id'] == task_id:
                    break
        else:
            img_id = random.choice(loader.img_ids)
            task_ix = random.choice(len(loader.ImgReferTasks[img_id]))
            task_id = loader.ImgReferTasks[task_ix]['task_id']

        img_data = loader.get_img_ref_data(img_id)
        p_structure = img_data['p_structures'][task_ix]
        gt_Polygons = img_data['gt_Polygons'][task_ix]
        gt_polygons = []
        for ps in gt_Polygons:
            gt_polygons += ps

        # 00: gt with gt mask and box
        gt_boxes = img_data['gt_boxes'][task_ix]
        phrase = img_data['phrases'][task_ix]

        n_a_r = '|'.join(p_structure['attributes']) + '||' + p_structure['name'] + '|'.join(p_structure['relations'])
        visualize_refvg(axes[0][0], img_id=img_id, gt_polygons=gt_polygons, gt_boxes=gt_boxes, title=n_a_r,
                        set_colors=colors)

        # 01:
        h = img_data['height']
        w = img_data['width']
        mps = polygons_to_mask(gt_polygons, w, h)
        b = np.sum(mps > 0, axis=None)
        rsize = b * 1.0 / (w * h)

        subset_cond = get_subset(phrase, p_structure, gt_boxes, rsize)
        subsets = [k for k, v in subset_cond.items() if v]
        subset_str = ' '.join(subsets)
        vg_boxes = img_data['vg_boxes'][task_ix]
        visualize_refvg(axes[0][1], img_id=img_id, vg_boxes=vg_boxes, title=subset_str, set_colors=colors)

        # 02:
        vg_str = ''
        for vg_ann_id in img_data['vg_ann_ids'][task_ix]:
            ann_data = loader.vg_loader.Anns[vg_ann_id]
            vg_str += 'box %s:\n%s\n%s\n%s\n' % (vg_ann_id, '|'.join(ann_data['names']),
                                                 '|'.join(ann_data['attributes']),
                                                 '|'.join(r['predicate'] for r in ann_data['relations']))
        axes[0, 2].text(0.1, 0.15, pred_str, transform=axes[0, 2].transAxes, fontsize=8)
        axes[0, 2].set_axis_off()

        # 03:
        if can_box_pred is not None:
            can_boxes = can_box_pred[task_id]
            visualize_refvg(axes[0, 3], img_id=img_id, can_boxes=can_boxes, title=len(can_boxes), set_colors=colors)

        # 1x:
        i = 0
        for p_name, pred in pred_dict.items():
            pred_boxes = pred.get('pred_boxes', None)
            pred_mask_bin = pred.get('pred_mask', None)
            if pred_mask_bin is not None:
                pred_mask = np.unpackbits(pred_mask_bin)[:img_data['height'] * img_data['width']]\
                    .reshape((img_data['height'], img_data['width']))
            else:
                pred_mask = None
            iou = iou_polygons_masks(gt_polygons, [pred_mask])
            visualize_refvg(axes[1, i], img_id=img_id, pred_boxes=pred_boxes, pred_mask=pred_mask,
                            title='%s:%.4f' % (p_name, iou), set_colors=colors)
            i += 1
            if i == 4:
                break

        task_str = '%s-%s-%s' % (task_id, phrase, subsets)
        fig.suptitle(task_str)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(fig, dpi=300)
        plt.close(fig)
        print('visualized %d / %d' % (vis_i, vis_count))

    return


if __name__ == '__main__':

    split = 'miniv'
    if split == 'test':
        ppaths = {'upperbound': 'det_upperbound_0.01_50_det_test0',
                 'ours': 'ensemble_pred/ensemble_IN_obj_cat_msub_wsub_mloc_wloc_mrel_wrel_m_att_logits_soft0.60_test0',
                 'Matt': 'det_mattnet_pred_0.15_50_det_test0',
                 'RMI': 'rmi_pred_test0'}
        fname = 'test_2814.npy'
    else:  # miniv
        ppaths = {'upperbound': 'det_upperbound_0.1_10_gt_miniv0',
                  'ensemble': 'ensemble_pred_topk1.000000_miniv0'}
        fname = 'miniv_17.npy'
    preds = dict()
    for p_name, p_path in ppaths.items():
        p = os.path.join('output/eval_refvg', p_path, fname)
        preds[p_name] = np.load(p_path).item()

    do_visualize(preds, can_box_pred=ppaths['upperbound'], vis_count=10, split='miniv')


