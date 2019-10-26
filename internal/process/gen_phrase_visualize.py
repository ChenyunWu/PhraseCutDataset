from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image

from _dataset.utils.vg_loader import VGLoader
from _comprehend.tools.mattnet_opt import parse_opt
from _dataset.utils.iou import iou_box


def visualize_imgs(loader, imgs, sample_count=0):
    fig, axes = plt.subplots(len(imgs), 3, figsize=(9, 3 * len(imgs)))
    t_count = 0
    for i, img in enumerate(imgs):
        img_path = 'data/refvg/images/%d.jpg' % img['image_id']
        img_channels = PIL_Image.open(img_path)

        axes[i, 0].imshow(img_channels)
        axes[i, 1].imshow(img_channels)

        ann_ids = img['ann_ids']
        random.shuffle(ann_ids)
        ref_phrases = ''

        for obj in img['objects']:
            if obj['object_id'] in ann_ids:
                continue
            axes[i, 0].add_patch(Rectangle((obj['x'], obj['y']), obj['w'], obj['h'],
                                        fill=False, edgecolor='gray', linewidth=0.6))
        for j, ann_id in enumerate(ann_ids):
            ann = loader.Anns[ann_id]
            x, y, w, h = ann['box']
            axes[i, 0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='blue', linewidth=0.6))

        dataset = gen_phrases(loader, [img], sample_count)
        t_count += len(dataset)
        for d_i, data in enumerate(dataset):
            for j, ann_id in enumerate(data['ann_ids']):
                ann = loader.Anns[ann_id]
                x, y, w, h = ann['box']
                if d_i < 10:
                    axes[i, 0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=0.6))
                    str = data['phrase']
                    axes[i, 0].text(x=x, y=y, s=str, style='italic', size='xx-small',
                                 bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})
                else:
                    axes[i, 0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='salmon', linewidth=0.6))

            if d_i < 25:
                ref_phrase = data['phrase']
                structure = data['phrase_structure']
                mode = structure['type']
                clues = loader.phrase_struct_to_str(loader.Anns[data['ann_ids'][0]])
                ref_phrases += ref_phrase + ' | ' + clues + ' (' + mode + ')\n'
        if len(ann_ids) > 25:
            ref_phrases += '...'

        axes[i, 2].text(x=0.01, y=0.01, s=ref_phrases, style='italic', size='xx-small')

        axes[i, 0].tick_params(labelbottom=False, labelleft=False)
        axes[i, 1].tick_params(labelbottom=False, labelleft=False)
        axes[i, 2].tick_params(labelbottom=False, labelleft=False)
        axes[i, 0].set_title('#ann=%d, #f_ann=%d' % (len(img['ann_ids']), len(dataset)))
        axes[i, 1].set_title('img_id=%d, #obj=%d' % (img['image_id'], len(img['objects'])))
    # fig.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.01)
    plt.savefig('visualize_imgs.pdf')
    print('#%d annotations in total' % t_count)
    print(loader.gen_phrase_stat)


def gen_phrases(loader, imgs, sample_count=0):
    dataset = []
    for i, img in enumerate(imgs):
        img_id = img['image_id']
        img_url = loader.info_dict[img_id]['url']
        ann_ids = img['ann_ids']
        if sample_count > 0:
            s_ann_ids = sample_img_anns(loader, img_id, sample_count)
        else:
            s_ann_ids = ann_ids
        phrases = {}  # dict of phrase: data
        for j, ann_id in enumerate(s_ann_ids):  # Bug here! there can be duplicates in s_ann_ids
            ref_phrase, structure = loader.gen_phrase(ann_id, ann_ids)
            if ref_phrase not in phrases:
                data = {'image_id': img['image_id'], 'image_url': img_url, 'ann_ids': [ann_id], 'phrase': ref_phrase,
                        'phrase_structure': structure, 'phrase_mode': structure['type']}
                phrases[ref_phrase] = data
            else:
                phrases[ref_phrase]['ann_ids'].append(ann_id)

        # remove similar phrases
        phrase_list = phrases.keys()
        removed_ids = []
        for i1, p1 in enumerate(phrase_list):
            name1 = phrases[p1]['phrase_structure']['name']
            for i2, p2 in enumerate(phrase_list):
                if i2 in removed_ids: continue
                if i1 != i2 and p1 in p2:
                    name2 = phrases[p2]['phrase_structure']['name']
                    if name1 == name2:
                        phrases.pop(p1, None)
                        removed_ids.append(i1)
                        # print('remove phrase "%s" by "%s"' % (p1, p2))
                        break
        if i % 100 == 0:
            print("%d/%d: %d annotations" % (i, len(imgs), len(phrases)))
        for data in phrases.values():
            data['task_id'] = str(data['image_id']) + '__' + '_'.join([str(ann_id) for ann_id in data['ann_ids']])
            dataset.append(data)
    return dataset


# self.objects[obj['object_id']] = {'ann_id': obj['object_id'], 'image_id': img['image_id'],
#                                'box': [obj['x'], obj['y'], obj['w'], obj['h']],
#                                'names': obj['names'],
#                                'attributes': attributes,
#                                'synsets': obj['synsets'],
#                                'relations': []}
def sample_img_anns(loader, img_id, max_count=20):
    img = loader.Images[img_id]
    img_w = img['width']
    img_h = img['height']
    ann_ids = img['ann_ids']
    f_ann_ids = []
    f_size_ratios = []
    for ann_id in ann_ids:
        ann = loader.Anns[ann_id]
        # filter by size
        box = ann['box']  # x, y, w, h
        size_ratio = box[2] * box[3] * 1.0 / img_h / img_w
        if size_ratio < 0.01:
            continue
        # filter by overlap (iou)
        for f_id in f_ann_ids:
            f_box = loader.Anns[f_id]['box']
            iou = iou_box(box, f_box, xywh=True)
            if iou > 0.2:
                continue
        f_ann_ids.append(ann_id)
        f_size_ratios.append(size_ratio)

    if len(f_ann_ids) <= max_count:
        return f_ann_ids

    # sample from filtered
    # basic sample weight: sqrt(box_size)
    # for each new sample, weights for its same type x 0.2
    for i, r in enumerate(f_size_ratios):
        if r > 0.9:
            f_size_ratios[i] = 0
    s = np.array(f_size_ratios)
    s = np.minimum(s, 1 - s)
    weights = np.sqrt(np.clip(s, 0, 0.1))
    same_types = np.zeros([len(f_ann_ids)] * 2)
    for i, id1 in enumerate(f_ann_ids):
        for j, id2 in enumerate(f_ann_ids[i+1:]):
            if loader.is_same_category(loader.Anns[id1], loader.Anns[id2]):
                same_types[i, j] = same_types[j, i] = 1
    # no same type: directly sample all
    if np.sum(same_types) == 0:
        prob = weights / np.sum(weights)
        s_ann_ids = np.random.choice(f_ann_ids, p=prob, size=max_count)
        return s_ann_ids
    # sample one by one
    s_ann_ids = []
    for c in range(max_count):
        prob = weights / np.sum(weights)
        s_id = np.random.choice(f_ann_ids, p=prob)
        s_ann_ids.append(s_id)
        i = f_ann_ids.index(s_id)
        weights[i] = 0
        for j in range(len(f_ann_ids)):
            if same_types[i, j] > 0 and i != j:
                weights[j] *= 0.2
    assert len(s_ann_ids) == len(set(s_ann_ids))
    return s_ann_ids


def get_subset(input_path, output_path, size=500):
    with open(input_path, 'r') as f:
        dataset = json.load(f)
    count = 0
    img_ids = []
    subset = []
    for data in dataset:
        if data['image_id'] not in img_ids:
            this_img = [d for d in dataset if d['image_id'] == data['image_id']]
            subset += this_img
            count += len(this_img)
            print(count)
            img_ids.append(data['image_id'])
        if count >= size:
            break
    with open(output_path, 'w') as f:
        json.dump(subset, f)
    return subset


def main(args):
    opt = vars(args)
    opt['dataset'] = 'refvg'
    opt['splitBy'] = 'all'
    opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']

    loader = VGLoader()

    # val_images = [img for img in loader.images.values() if img['split'] == 'val']
    # visualize_imgs(loader, random.sample(val_images, 20))
    # dataset = gen_phrases(loader, val_images)
    # with open('data/refvg/phrases_val1000_temp.json', 'w') as f:
    #     json.dump(dataset, f)
    train_imgs = [img for img in loader.Images.values() if img['split'] == 'train']
    s_imgs = random.sample(train_imgs, 40)
    # if 1593062 in loader.images:
    #     s_imgs = [loader.images[1593062]] + s_imgs
    visualize_imgs(loader, s_imgs, sample_count=6)
    dataset = gen_phrases(loader, train_imgs, sample_count=6)
    # with open('data/refvg/phrases_train1000.json', 'w') as f:
    #     json.dump(dataset, f)


if __name__ == '__main__':

    args = parse_opt()
    main(args)

    # get_subset('data/refvg/phrases_val.json', 'data/refvg/phrases_val_sub100.json', 100)