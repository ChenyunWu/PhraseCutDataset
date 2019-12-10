import os

dataset_dir = 'data/VGPhraseCut_v0'

img_fpath = os.path.join(dataset_dir, 'images')
name_att_rel_count_fpath = os.path.join(dataset_dir, 'name_att_rel_count.json')
img_info_fpath = os.path.join(dataset_dir, 'image_data_split3000.json')
skip_fpath = os.path.join(dataset_dir, 'skip.json')

refer_fpaths = dict()
vg_scene_graph_fpaths = dict()
for split in ['train', 'val', 'test', 'miniv']:
    refer_fpaths[split] = os.path.join(dataset_dir, 'refer_%s.json' % split)
    vg_scene_graph_fpaths[split] = os.path.join(dataset_dir, 'refer_%s.json' % split)
