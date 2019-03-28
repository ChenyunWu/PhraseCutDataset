import requests
import StringIO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

plt.switch_backend('agg')

from data_transfer import xyxy_to_xywh
from iou import polygons_to_mask

# color for "pred_mask" should be a colormap
# color for "gt_polygons" can be "colorful": different color for each polygon
visualize_colors = {'title': 'black', 'gt_polygons': 'deepskyblue', 'gt_boxes': 'blue', 'gt_all_boxes': 'blue',
                    'vg_boxes': 'green', 'vg_all_boxes': 'green', 'pred_boxes': 'red', 'pred_mask': 'autumn',
                    'can_boxes': 'red'}


def visualize_refvg(ax, img_id=-1, img_url=None, title=None, gt_Polygons=None, gt_polygons=None, gt_boxes=None,
                    gt_all_boxes=None, vg_boxes=None, vg_all_boxes=None, pred_boxes=None, pred_mask=None,
                    can_boxes=None, set_colors=None, xywh=True):
    """
    Plot the image in ax and the provided annotations. boxes are lists of [x1, y1, x2, y2].
    Draw less important things first.
    :param ax:
    :param img_id: if > 0, get img from local path
    :param img_url: if img_id<=0, get img from this url. Must set either img_id or img_url
    :param title: plot title. better to put referring phrases here
    :param gt_Polygons: Polygons from AMT. list of {'instance_id': id, 'points': [[x1, y1], [x2, y2], ...]}
    :param gt_polygons: Only used when gt_Polygons=None [(polygon:)[(point:)[x1, y1], [x2, y2], ...], [[],[],...],...]
    :param gt_boxes: gt instance boxes from AMT polygons.[[x1, y1, x2, y2],[],...]
    :param gt_all_boxes: all instance boxes from AMT polygons in this img
    :param vg_boxes: vg boxes used to generate the phrase
    :param vg_all_boxes: all boxes from VG (after filtering)
    :param pred_boxes: predicted boxes
    :param pred_mask: predicted mask. 2D binary numpy array the same shape as the img
    :return: Nothing
    """
    def modify_color(d):
        colors = visualize_colors
        if not d:
            return colors
        for name, color in d.items():
            colors[name] = color
        return colors

    colors = modify_color(set_colors)

    if img_id < 0 and not img_url:
        return

    # show img
    if img_id:
        img = Image.open('data/refvg/images/%d.jpg' % img_id)
    else:
        response = requests.get(img_url, verify=False)
        img = Image.open(StringIO(response.content))
    ax.imshow(img)

    if not xywh:
        for boxes in [gt_boxes, gt_all_boxes, vg_boxes, vg_all_boxes, pred_boxes]:
            if boxes:
                boxes[:] = xyxy_to_xywh(boxes)

    if vg_all_boxes:
        for box in vg_all_boxes:
            ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=colors['vg_all_boxes'],
                                   linewidth=0.3, linestyle=':', alpha=0.5))
    if gt_all_boxes:
        for box in gt_all_boxes:
            ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=colors['gt_all_boxes'],
                                   linewidth=0.3, linestyle=':', alpha=0.5))

    color = colors['gt_polygons']
    if gt_Polygons is not None:
        for ins_i, ins_ps in enumerate(gt_Polygons):
            if color == 'colorful':
                c = 'C%d' % (ins_i % 10)
            else:
                c = color
            mps = polygons_to_mask(ins_ps, img.size[1], img.size[0])
            masked = np.ma.masked_where(mps == 0, mps)
            ax.imshow(masked, 'Greens_r', interpolation='none', alpha=0.6)
            for p in ins_ps:
                ax.add_patch(Polygon(p, fill=True, alpha=0.5, color=c))
    elif gt_polygons is not None:
        for pi, polygon in enumerate(gt_polygons):
            if color == 'colorful':
                c = 'C%d' % (pi % 10)
            else:
                c = color
            ax.add_patch(Polygon(polygon, fill=True, alpha=0.5, color=c))

    if vg_boxes is not None:
        for box in vg_boxes:
            ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=colors['vg_boxes'],
                                   linewidth=0.9, linestyle='-', alpha=0.8))
    if gt_boxes is not None:
        for box in gt_boxes:
            ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=colors['gt_boxes'],
                                   linewidth=0.9, linestyle='-', alpha=0.8))

    if can_boxes is not None:
        for box in can_boxes:
            ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=colors['can_boxes'],
                                   linewidth=0.6, linestyle=':', alpha=0.8))

    if pred_boxes is not None:
        for box in pred_boxes:
            ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=colors['pred_boxes'],
                                   linewidth=0.6, linestyle='-', alpha=0.9))
    if pred_mask is not None:
        masked = np.ma.masked_where(pred_mask == 0, pred_mask)
        ax.imshow(masked, colors['pred_mask'], interpolation='none', alpha=0.7)

    # ax.set_frame_on(False)
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    ax.set_axis_off()

    if title:
        ax.set_title(title, color=colors['title'])
    return
