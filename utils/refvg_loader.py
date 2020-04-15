import json
import random
import numpy as np

from phrase_handler import PhraseHandler
from vg_loader import VGLoader
from subset import get_subset
from data_transfer import polygons_to_mask
from file_paths import img_info_fpath, refer_fpaths, refer_input_fpaths


class RefVGLoader(object):

    def __init__(self, split=None, phrase_handler=None, word_embed=None, allow_no_att=True, allow_no_rel=True,
                 include_vg_scene_graph=False, input_anno_only=False):

        if phrase_handler is None:
            phrase_handler = PhraseHandler(word_embed=word_embed)
        self.phrase_handler = phrase_handler

        self.vg_loader = None
        if include_vg_scene_graph:
            assert not input_anno_only
            self.vg_loader = VGLoader(split=split, phrase_handler=phrase_handler, obj_filter=False)

        ref_tasks = []
        if not split:
            self.splits = ['miniv', 'val', 'test', 'train']
        else:
            self.splits = split.split('_')

        print('RefVGLoader loading img_info: %s' % img_info_fpath)
        with open(img_info_fpath, 'r') as f:
            imgs_info = json.load(f)
            self.ImgInfo = {img['image_id']: img for img in imgs_info if img['split'] in self.splits}

        print('RefVGLoader loading refer data')

        for s in self.splits:
            if input_anno_only:
                fpath = refer_input_fpaths[s]
            else:
                fpath = refer_fpaths[s]
            print('RefVGLoader loading %s' % fpath)
            with open(fpath, 'r') as f:
                ref_tasks += json.load(f)

        print('RefVGLoader preparing data')
        self.task_num = 0
        self.ImgReferTasks = dict()
        self.ImgInsBoxes = dict()
        self.ImgInsPolygons = dict()
        for task in ref_tasks:
            if not allow_no_att and len(task['phrase_structure']['attributes']) == 0:
                continue
            if not allow_no_rel and len(task['phrase_structure']['relation_descriptions']) == 0:
                continue
            img_id = task['image_id']
            self.ImgReferTasks[img_id] = self.ImgReferTasks.get(img_id, list()) + [task]
            if not input_anno_only:
                self.ImgInsBoxes[img_id] = self.ImgInsBoxes.get(img_id, list()) + task['instance_boxes']
                self.ImgInsPolygons[img_id] = self.ImgInsPolygons.get(img_id, list()) + task['Polygons']
                task['ins_box_ixs'] = range(len(self.ImgInsBoxes[img_id]) - len(task['instance_boxes']),
                                            len(self.ImgInsBoxes[img_id]))
            self.task_num += 1
        self.img_ids = list(self.ImgReferTasks.keys())
        self.shuffle()
        self.iterator = 0
        self.input_anno_only = input_anno_only
        print('split %s: %d imgs, %d tasks' % ('_'.join(self.splits), len(self.img_ids), self.task_num))
        print('RefVGLoader ready.')

    def shuffle(self):
        random.shuffle(self.img_ids)
        return

    def get_task_subset(self, img_id, task_id):
        img_info = self.ImgInfo[img_id]
        task = None
        for t in self.ImgReferTasks[img_id]:
            if t['task_id'] == task_id:
                task = t
                break
        if 'subsets' in task:
            return task['subsets']

        polygons = list()
        for ps in task['Polygons']:
            polygons += ps
        mps = polygons_to_mask(polygons, img_info['width'], img_info['height'])
        b = np.sum(mps > 0, axis=None)
        gt_relative_size = b * 1.0 / (img_info['width'] * img_info['height'])
        cond = get_subset(task['image_id'], task['phrase_structure'], task['instance_boxes'], gt_relative_size)
        task['subsets'] = [k for k, v in cond.items() if v]
        return task['subsets']

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

        phrases = []
        task_ids = []
        p_structures = []
        gt_Polygons = []
        gt_boxes = []
        img_ins_cats = []
        img_ins_atts = []
        for task in self.ImgReferTasks[img_id]:
            phrases.append(task['phrase'])
            task_ids.append(task['task_id'])
            p_structures.append(task['phrase_structure'])
            if not self.input_anno_only:
                gt_boxes.append(task['instance_boxes'])
                gt_Polygons.append(task['Polygons'])
                img_ins_cats += [task['phrase_structure']['name']] * len(task['instance_boxes'])
                img_ins_atts += [task['phrase_structure']['attributes']] * len(task['instance_boxes'])

        # return data
        data = dict()
        data['image_id'] = img_id
        img_info = self.ImgInfo[img_id]
        data['width'] = img_info['width']
        data['height'] = img_info['height']
        data['split'] = img_info['split']

        data['task_ids'] = task_ids
        data['phrases'] = phrases
        data['p_structures'] = p_structures
        if not self.input_anno_only:
            data['img_ins_boxes'] = self.ImgInsBoxes[img_id]
            data['img_ins_Polygons'] = self.ImgInsPolygons[img_id]
            data['img_ins_cats'] = img_ins_cats
            data['img_ins_atts'] = img_ins_atts
            data['gt_Polygons'] = gt_Polygons
            data['gt_boxes'] = gt_boxes

            if self.vg_loader is not None:
                vg_img = self.vg_loader.images[img_id]
                vg_obj_ids = []
                vg_boxes = []

                img_obj_ids = vg_img['obj_ids']
                img_vg_boxes = [self.vg_loader.objects[obj_id]['box'] for obj_id in set(img_obj_ids)]

                for task in self.ImgReferTasks[img_id]:
                    task_obj_ids = [i for i in task['ann_ids'] if i in img_obj_ids]
                    vg_obj_ids.append(task_obj_ids)
                    vg_boxes.append([self.vg_loader.objects[obj_id]['box'] for obj_id in set(task_obj_ids)])

                data['img_vg_boxes'] = img_vg_boxes
                data['vg_boxes'] = vg_boxes
                data['vg_obj_ids'] = vg_obj_ids

        data['bounds'] = {'it_pos_now': self.iterator, 'it_max': max_index, 'wrapped': wrapped}

        return data
