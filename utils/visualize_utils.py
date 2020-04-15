# import StringIO
# import requests
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

plt.switch_backend('agg')

from data_transfer import xyxy_to_xywh
from file_paths import img_fpath


# color for "pred_mask" should be a colormap
# color for "gt_polygons" can be "colorful": different color for each polygon
# visualize_colors = {'title': 'black', 'gt_mask': 'Blues', 'gt_polygons': 'deepskyblue', 'gt_boxes': 'blue',
#                     'gt_all_boxes': 'blue', 'vg_boxes': 'green', 'vg_all_boxes': 'green', 'pred_boxlist': 'red',
#                     'pred_mask': 'autumn', 'can_boxes': 'red'}

visualize_colors = {'title': 'black', 'gt_mask': 'Wistia', 'gt_polygons': 'dodgerblue', 'gt_boxes': 'royalblue',
                    'gt_all_boxes': 'gold', 'vg_boxes': 'green', 'vg_all_boxes': 'green', 'pred_boxlist': 'deepskyblue',
                    'pred_mask': 'GnBu', 'pred_scores': 'cividis', 'can_boxes': 'darkcyan'}


def plot_refvg(ax=None, fig=None, fig_size=None, img=None, img_id=-1, img_url=None, title=None, font_size=5,
               gt_mask=None, gt_Polygons=None, gt_polygons=None, gt_boxes=None, gt_all_boxes=None, vg_boxes=None,
               vg_all_boxes=None, pred_boxes=None, pred_mask=None, pred_scores=None, can_boxes=None, set_colors=None,
               xywh=True, cbar=None, gray_img=True, img_hw=None, range01=True):
    """
    Plot the image in ax and the provided annotations. Draw less important things first.
    :param ax:
    :param fig: only needed for creating color bar
    :param fig_size: if both ax and fig are None, create ax and fig by this size
    :param img: PIL image
    :param img_id: if > 0, get img from local path
    :param img_url: if img_id<=0, get img from this url. Must set one of img, img_id or img_url
    :param title: plot title. better to put referring phrases here
    :param font_size: fontsize for title
    :param gt_mask: 2D binary numpy array the same shape as the img
    :param gt_Polygons: Polygons from AMT. list of instance polygons
    :param gt_polygons: Only used when gt_Polygons=None [(polygon:)[(point:)[x1, y1], [x2, y2], ...], [[],[],...],...]
    :param gt_boxes: gt instance boxes from AMT polygons.[[x1, y1, x2, y2],[],...]
    :param gt_all_boxes: all instance boxes from AMT polygons in this img
    :param vg_boxes: vg boxes used to generate the phrase
    :param vg_all_boxes: all boxes from VG (after filtering)
    :param pred_boxes: predicted boxes
    :param pred_mask: predicted binary mask. 2D binary numpy array, the same shape as the img or img_hw
    :param pred_scores: predicted score mask. 2D float numpy array, the same shape as the img or img_hw
    :param can_boxes: candidate boxes
    :param set_colors: change default color by a dict
    :param xywh: whether input boxes are xywh or xyxy
    :param cbar: which color bar to show. None/'gt'/'pred'. only used when gt_mask/pred_mask/pred_scores is provided.
    :param gray_img: whether to make the background image grayscale.
    :param img_hw: if not None, reshape the image to this size. Use only if pred_mask/scores are not in original size.
                (May make the box scales wrong)
    :param range01: Only for pred_scores. Whether to force the range of color bar to 0 ~ 1.
    :return: fig
    """
    def modify_color(d):
        colors = visualize_colors.copy()
        if d is None:
            return colors
        for name, color in d.items():
            colors[name] = color
        return colors

    colors = modify_color(set_colors)

    if img_id < 0 and img_url is None and img is None:
        return
    if ax is None and fig is None:
        if fig_size is not None:
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title, color=colors['title'], fontsize=font_size)

    # show img
    if img is None:
        if img_id >= 0:
            img = Image.open(os.path.join(img_fpath, '%d.jpg' % img_id))
        # elif img_url is not None:
        #     response = requests.get(img_url, verify=False)
        #     img = Image.open(StringIO(response.content))
            if img_hw is not None:
                ori_w, ori_h = img.size
                if ori_w != img_hw[1] or ori_h != img_hw[0]:
                    img = img.resize((img_hw[1], img_hw[0]))
        else:
            raise NotImplementedError

    if gray_img:
        image = img.convert("L")
        arr = np.asarray(image)
        plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    else:
        ax.imshow(img)

    if not xywh:
        for boxes in [gt_boxes, gt_all_boxes, vg_boxes, vg_all_boxes, pred_boxes]:
            if boxes is not None:
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
    if gt_mask is not None:
        masked = np.ma.masked_where(gt_mask == 0, gt_mask)
        p = ax.imshow(masked, colors['gt_mask'], interpolation='none', alpha=0.5, vmin=0, vmax=1.0)
        if cbar == 'gt':
            cb = fig.colorbar(p, ax=ax, format='%.1f')
            # cb.ax.tick_params(labelsize=5)
    elif gt_Polygons is not None:
        for ins_i, ins_ps in enumerate(gt_Polygons):
            if color == 'colorful':
                c = 'C%d' % (ins_i % 10)
            else:
                c = color
            # mps = polygons_to_mask(ins_ps, img.size[1], img.size[0])
            # masked = np.ma.masked_where(mps == 0, mps)
            # ax.imshow(masked, 'Greens_r', interpolation='none', alpha=0.6)
            for p in ins_ps:
                ax.add_patch(Polygon(p, fill=True, alpha=0.9, color=c))
    elif gt_polygons is not None:
        for pi, polygon in enumerate(gt_polygons):
            if color == 'colorful':
                c = 'C%d' % (pi % 10)
            else:
                c = color
            ax.add_patch(Polygon(polygon, fill=True, alpha=0.9, color=c))

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
            ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=colors['pred_boxlist'],
                                   linewidth=0.6, linestyle='-', alpha=0.9))
    if pred_mask is not None:
        masked = np.ma.masked_where(pred_mask == 0, pred_mask)
        ax.imshow(masked, colors['pred_mask'], interpolation='none', alpha=0.9, vmin=0, vmax=1.2)

    if pred_scores is not None:
        if range01:
            p = ax.imshow(pred_scores, colors['pred_scores'], interpolation='none', alpha=1.0, vmin=0, vmax=1.0)
        else:
            p = ax.imshow(pred_scores, colors['pred_scores'], interpolation='none', alpha=1.0)
        if cbar == 'pred':
            cb = fig.colorbar(p, ax=ax, format='%.1f')
            # cb.ax.tick_params(labelsize=4)

    ax.set_frame_on(False)
    # # ax.set_axis_off() --> DON'T USE THIS! Will still leave blank space for axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig


def gt_visualize_to_file(img_data, task_id, fig_path, skip_exist=True, gray_img=True):
    img_id = img_data['image_id']
    fig_h = img_data['height'] / 300.0
    fig_w = img_data['width'] / 300.0
    fig_dir = os.path.dirname(fig_path)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if os.path.exists(fig_path) and skip_exist:
        return False
    task_i = img_data['task_ids'].index(task_id)
    gt_Polygons = img_data['gt_Polygons'][task_i]
    gt_boxes = img_data['gt_boxes'][task_i]
    try:
        fig = plot_refvg(fig_size=[fig_w, fig_h], img_id=img_id, gt_Polygons=gt_Polygons, gt_boxes=gt_boxes,
                         gray_img=gray_img)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e:
        print('WARNING: gt_visualize_to_file fail on %s' % task_id)
        print(e)
        print(fig_path)
        print('img_size:', img_data['height'], img_data['width'])
        print('fig_size:', fig_h, fig_w)
        print('gt_Polygons:', gt_Polygons)
        print('gt_boxes:', gt_boxes)
    return True


def pred_visualize_to_file(img_data, fig_path, pred_boxlist=None, pred_boxes=None, pred_mask=None, can_boxes=None,
                           skip_exist=True, xywh=True):
    img_id = img_data['image_id']
    fig_h = img_data['height'] / 300.0
    fig_w = img_data['width'] / 300.0
    if os.path.exists(fig_path) and skip_exist:
        return False
    try:
        boxes = None
        img_hw = None
        if pred_boxlist is not None:
            pred_boxes = pred_boxlist.bbox.cpu().numpy(),
            img_hw = (pred_boxlist.size[1], pred_boxlist.size[0])
            xywh = pred_boxlist.mode == 'xywh'
        fig = plot_refvg(fig_size=[fig_w, fig_h], img_id=img_id, pred_boxes=pred_boxes, pred_mask=pred_mask,
                         can_boxes=can_boxes, xywh=xywh, gray_img=True, img_hw=img_hw)
        fig.savefig(fig_path, dpi=300,  bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e:
        print('WARNING: pred_visualize_to_file fail on %s' % img_id)
        print(e)
        print('img_size:', img_data['height'], img_data['width'])
        print('fig_size:', fig_h, fig_w)
    return True


def score_visualize_to_file(img_data, fig_path, score_mask, skip_exist=True, include_cbar=True, range01=True):
    img_id = img_data['image_id']
    fig_h = img_data['height'] / 300.0
    fig_w = img_data['width'] / 300.0
    if include_cbar:
        fig_w += fig_h * 0.2
    if os.path.exists(fig_path) and skip_exist:
        return False
    try:
        cbar = ''
        if include_cbar:
            cbar = 'pred'
        fig = plot_refvg(fig_size=[fig_w, fig_h], img_id=img_id, pred_scores=score_mask, cbar=cbar, gray_img=True,
                         img_hw=score_mask.shape, range01=range01)  # viridis
        fig.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e:
        print('WARNING: score_visualize_to_file fail on %s' % img_id)
        print(e)
        print('img_size:', img_data['height'], img_data['width'])
        print('fig_size:', fig_h, fig_w)
    return True


def save_pred_to_png(pred_mask, fig_path):
    # print('np hw', pred_mask.shape)
    # img = Image.fromarray(pred_mask.astype(int), 'L').convert('1')
    h, w = pred_mask.shape
    img = Image.new('1', (w, h))
    pixels = img.load()
    for i in range(w):
        for j in range(h):
            pixels[i, j] = int(pred_mask[j, i])
    # print('pil wh', img.size)
    img.save(fig_path)


def png_to_pred_mask(png_path):
    im_frame = Image.open(png_path)
    w, h = im_frame.size
    pred_mask = np.array(im_frame.getdata()).reshape((h, w))
    return pred_mask
