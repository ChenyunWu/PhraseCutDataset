from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random

from .vg_loader import VGLoader as VGLoader


class RefVGLoader:

    def __init__(self, split=None, rebalance=False, vg_loader=None, obj_filter=False, allow_no_structure=False,
                 word_embed=None, allow_no_att=True):

        self.vg_loader = vg_loader
        if not vg_loader:
            self.vg_loader = VGLoader(split=split, word_embed=word_embed, obj_filter=obj_filter)

        ref_tasks = []
        if not split:
            self.splits = ['miniv', 'val', 'test', 'train']
        else:
            self.splits = split.split('_')

        print('RefVGLoader loading refer data')
        for s in self.splits:
            if rebalance:
                print('RefVGLoader loading refer_filtered_instance_rebalance_%s.json' % s)
                with open('data/refvg/amt_result/refer_filtered_instance_rebalance_%s.json' % s, 'r') as f:
                    ref_tasks += json.load(f)
            else:
                print('RefVGLoader loading refer_filtered_instance_%s.json' % s)
                with open('data/refvg/amt_result/refer_filtered_instance_%s.json' % s, 'r') as f:
                    ref_tasks += json.load(f)

        print('RefVGLoader preparing data')
        self.ImgInsBoxes = {}
        self.ImgInsPolygons = {}
        self.ImgReferTasks = {}
        for task in ref_tasks:
            if not allow_no_structure and not task['phrase_structure']:
                continue
            if not allow_no_att and len(task['phrase_structure']['attributes']) == 0:
                continue
            img_id = task['image_id']
            # if 'relations' in task['phrase_structure']:
            #     task['phrase_structure']['rel_descriptions'] = self.get_rel_descriptions(task['phrase'],
            #                                                                              task['phrase_structure'])
            if img_id in self.ImgReferTasks.keys():
                self.ImgReferTasks[img_id].append(task)
                self.ImgInsBoxes[img_id] += task['instance_boxes']
                self.ImgInsPolygons[img_id] += task['Polygons']
                task['ins_box_ixs'] = range(len(self.ImgInsBoxes[img_id]) - len(task['instance_boxes']),
                                            len(self.ImgInsBoxes[img_id]))
            else:
                self.ImgReferTasks[img_id] = [task]
                self.ImgInsBoxes[img_id] = task['instance_boxes'][:]
                self.ImgInsPolygons[img_id] = task['Polygons'][:]
                task['ins_box_ixs'] = range(len(task['instance_boxes']))

        self.img_ids = list(self.ImgInsBoxes.keys())
        self.shuffle()
        self.iterator = 0
        print('RefVGLoader ready.')

    def shuffle(self):
        random.shuffle(self.img_ids)
        return

    def get_img_ref_data(self, img_id=-1):
        """
        get a batch with one image and all refer data on that image
        """
        # Fetch feats according to the image_split_ix
        wrapped = False
        max_index = len(self.img_ids) - 1

        if img_id < 0:
            ri = self.iterator
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.iterator = ri_next
            img_id = self.img_ids[ri]

        vg_img = self.vg_loader.images[img_id]
        vg_obj_id_set = set()
        vg_obj_ids = []
        vg_boxes = []

        img_obj_ids = vg_img['obj_ids']
        img_vg_boxes = [self.vg_loader.objects[obj_id]['box'] for obj_id in img_obj_ids]
        img_ins_boxes = self.ImgInsBoxes[img_id]
        img_ins_Polygons = self.ImgInsPolygons[img_id]
        for task in self.ImgReferTasks[img_id]:
            # for i in task['ann_ids']:
                # if i not in img_ann_ids:
                #     print('obj_id not in vg_pp: %s' % i)
            task_obj_ids = [i for i in task['ann_ids'] if i in img_obj_ids]
            vg_obj_ids.append(task_obj_ids)
            vg_obj_id_set.update(task_obj_ids)
            vg_boxes.append([self.vg_loader.objects[obj_id]['box'] for obj_id in task_obj_ids])

        phrases = []
        task_ids = []
        p_structures = []
        gt_Polygons = []
        gt_boxes = []
        img_ins_cats = []
        img_ins_atts = []
        # img_ins_rels = []
        for task in self.ImgReferTasks[img_id]:
            phrases.append(task['phrase'])
            task_ids.append(task['task_id'])
            gt_boxes.append(task['instance_boxes'])
            p_structures.append(task['phrase_structure'])
            gt_Polygons.append(task['Polygons'])
            img_ins_cats += [task['phrase_structure']['name']] * len(task['instance_boxes'])
            img_ins_atts += [task['phrase_structure']['attributes']] * len(task['instance_boxes'])
            # img_ins_rels += [task['phrase_structure']['relations']] * len(task['instance_boxes'])

        # return data
        data = dict()
        data['image_id'] = img_id
        data['width'] = vg_img['width']
        data['height'] = vg_img['height']
        data['split'] = vg_img['split']

        data['task_ids'] = task_ids
        data['phrases'] = phrases
        data['p_structures'] = p_structures

        data['img_vg_boxes'] = img_vg_boxes
        data['img_ins_boxes'] = img_ins_boxes
        data['img_ins_Polygons'] = img_ins_Polygons
        data['img_ins_cats'] = img_ins_cats
        data['img_ins_atts'] = img_ins_atts
        # data['img_ins_rels'] = img_ins_rels
        data['gt_Polygons'] = gt_Polygons
        data['gt_boxes'] = gt_boxes
        data['vg_boxes'] = vg_boxes
        data['vg_obj_ids'] = vg_obj_ids

        data['bounds'] = {'it_pos_now': self.iterator, 'it_max': max_index, 'wrapped': wrapped}

        return data

