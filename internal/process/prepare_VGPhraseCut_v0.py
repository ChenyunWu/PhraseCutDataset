import json
import random


def slim_ref():
    """
    Format of refer_filtered_instance_refine_xxx.json:
    List of dicts ('task_id', 'image_id', 'ann_ids', 'Polygons', 'instance_boxes', 'phrase', 'phrase_structure',
                    'iou', 'iob', 'iop', 'turk_id')
    - task_id: unique id for each phrase-region pair (constructed from image_id and ann_ids)
    - image_id: image id from Visual Genome
    - ann_ids: all object ids (in Visual Genome) that match the phrase
    - instance_boxes: list of referred instance boxes (xywh format)
    - Polygons: list of "instance_polygons", same length as instance_boxes. "instance_polygons":
        list of "polygon"s for a single instance. "polygon": list of [x, y] points, representing one polygon.
    - phrase: the referring phrase as a string
    - phrase_structure: dict of ('name', 'attributes', 'relations', 'relation_descriptions', 'type')
            - relations: list of dict, same as relations from VG: {'subject_id': ann_id (in VG) of the target,
                should be the same as the first one in ann_ids; 'predicate': string for the relation predicate;
                'object_id': ann_id (in VG) of the supporting object; 'relationship_id': from VG;
                'synsets': of the predicate, from VG}
            - relation_descriptions: list of relation_description. each relation_description is a list of two elements:
                string for the predicate, string (category name) for the supporting object
            - type: name (name is unique), attribute (att+name is unique), relation (name+relation is unique),
            verbose (not unique)
    - iou / iop / turk_id: used only for filtering the data

    After slim:
    List of dicts ('task_id', 'image_id', 'ann_ids', 'Polygons', 'instance_boxes', 'phrase', 'phrase_structure')
    - phrase_structure: dict of ('name', 'attributes', 'relation_ids', 'relation_descriptions', 'type')
    """
    # for split in ['miniv', 'val', 'test', 'train']:
    fpath = 'data/refvg/amt_result/refer_filtered_instance_refine.json'
    ref = json.load(open(fpath))
    for task in ref:
        for k in ['iou', 'iop', 'turk_id']:  # missed iob here!
            if k in task:
                del task[k]
        ps = task['phrase_structure']
        rids = [r['relationship_id'] for r in ps['relations']]
        ps['relation_ids'] = rids
        del ps['relations']

    new_path = 'data/refvg/amt_result/refer_filtered_instance_refine_slim.json'
    with open(new_path, 'w') as f:
        json.dump(ref, f)
    print('%s saved.' % new_path)


def slim_ref2():
    """
    remove iob, original_phrase
    """
    print('slim_ref2')
    fpath = 'data/refvg/amt_result/refer_filtered_instance_refine_slim.json'
    ref = json.load(open(fpath))
    for task in ref:
        for k in ['original_phrase']:
            if k in task:
                print('original vs. phrase: "%s" "%s"' % (task['original_phrase'], task['phrase']))

                del task[k]

    new_path = 'data/refvg/amt_result/refer_filtered_instance_refine_slim.json'
    with open(new_path, 'w') as f:
        json.dump(ref, f)
    print('%s saved.' % new_path)


def slim_skip():
    """
    Format of skip_vxx.json:
    List of dicts ('task_id', 'image_id', 'ann_ids', 'skip_reason', 'phrase', 'phrase_structure', 'turk_id', 'boxes')
    - phrase_structure: dict of ('name', 'attributes', 'relations', 'type')
            - relations: list of dict, same as relations from VG: {'subject_id': ann_id (in VG) of the target,
                should be the same as the first one in ann_ids; 'predicate': string for the relation predicate;
                'object_id': ann_id (in VG) of the supporting object; 'relationship_id': from VG;
                'synsets': of the predicate, from VG}
              - type: name (name is unique), attribute (att+name is unique), relation (name+relation is unique),
            verbose (not unique)

    After slim:
    remove 'turk_id', replace 'relations' with 'relation_ids' in phrase_structure
    List of dicts ('task_id', 'image_id', 'ann_ids', 'skip_reason', 'phrase', 'phrase_structure', 'boxes')
    - phrase_structure: dict of ('name', 'attributes', 'relation_ids', 'type')
    """
    skipped = list()
    for v in ['v1.01', 'v1.2', 'v2']:
        fpath = 'data/refvg/amt_result/skip_%s.json' % v
        skipped += json.load(open(fpath))

    for task in skipped:
        del task['turk_id']
        if 'phrase_structure' in task:
            ps = task['phrase_structure']
            rids = [r['relationship_id'] for r in ps['relations']]
            ps['relation_ids'] = rids
            del ps['relations']

    print(len(skipped))
    new_path = 'data/refvg/amt_result/skip_slim.json'
    with open(new_path, 'w') as f:
        json.dump(skipped, f)
    print('%s saved.' % new_path)


def update_split():
    """
    Update image_data_split3000.json and scene_graphs_train.json:
        - remove train images that are not annotated (include only images in refer_train and skip_train)
        - larger miniv with 100 images from val
    """
    # remove train images that are not annotated
    fpath = 'data/VGPhraseCut_v0/refer_train.json'
    ref = json.load(open(fpath))
    train_ids = set()
    for task in ref:
        train_ids.add(task['image_id'])

    skip_f = 'data/VGPhraseCut_v0/skip.json'
    skip = json.load(open(skip_f))
    skip_ids = set()
    for task in skip:
        skip_ids.add(task['image_id'])

    keep_ids = train_ids.union(skip_ids)

    with open('data/refvg/image_data_split3000.json') as f:
        info = json.load(f)
    to_remove = list()
    for img in info:
        if img['split'] == 'train' and img['image_id'] not in keep_ids:
            to_remove.append(img['image_id'])

    new_info = [img for img in info if img['image_id'] not in to_remove]
    print(len(info), len(new_info))

    val_ids = [img['image_id'] for img in new_info if img['split'] in ['val', 'miniv']]
    miniv_ids = random.sample(val_ids, 100)
    for img in new_info:
        if img['image_id'] in val_ids:
            img['split'] = 'val'
        if img['image_id'] in miniv_ids:
            img['split'] = 'miniv'

    with open('data/refvg/image_data_split3000_100_slim.json', 'w') as f:
        json.dump(new_info, f)
    print('data/refvg/image_data_split3000_100_slim.json saved.')


def split_scene_graph():
    raw_file = '../data/VisualGenome1.2/scene_graphs.json'
    info_file = 'data/refvg/image_data_split3000_100_slim_nococo.json'
    splits = ['miniv', 'test', 'val', 'train']

    info = json.load(open(info_file))
    id_to_split = {img['image_id']: img['split'] for img in info}
    print(len(id_to_split))

    rel = json.load(open(raw_file))
    rels = dict()
    for s in splits:
        rels[s] = list()
    for r in rel:
        img_id = r['image_id']
        if img_id not in id_to_split:
            continue
        s = id_to_split[img_id]
        rels[s].append(r)

    for k, v in rels.items():
        out_fpath = 'data/VGPhraseCut_v0/scene_graphs_%s.json' % k
        with open(out_fpath, 'w') as f:
            json.dump(v, f)
        print(k, len(v), 'saved to %s' % out_fpath)


def split_ref(ref_pre='data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup'):
    print('split_ref')
    raw_file = ref_pre + '.json'
    info_file = 'data/refvg/image_data_split3000_100_slim.json'
    splits = ['miniv', 'test', 'val', 'train']

    info = json.load(open(info_file))
    id_to_split = {img['image_id']: img['split'] for img in info}
    print(len(id_to_split))

    rel = json.load(open(raw_file))
    rels = dict()
    for s in splits:
        rels[s] = list()
    for r in rel:
        img_id = r['image_id']
        s = id_to_split[img_id]
        rels[s].append(r)

    for k, v in rels.items():
        out_fpath = ref_pre + '_%s.json' % k
        with open(out_fpath, 'w') as f:
            json.dump(v, f)
        print(k, len(v), 'saved to %s' % out_fpath)


def remove_ref_duplicate():
    print('start of remove_ref_duplicate')
    old_file = 'data/refvg/amt_result/refer_filtered_instance_refine_slim.json'
    new_file = 'data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup.json'
    with open(old_file) as f:
        refs = json.load(f)

    task_ids = set()
    dup_ids = list()
    kept = list()

    for ref in refs:
        if ref['task_id'] in task_ids:
            dup_ids.append(ref['task_id'])
            print('DUPLICATE %d: %s' % (len(dup_ids), ref['task_id']))
        else:
            task_ids.add(ref['task_id'])
            kept.append(ref)

    print('duplicate count: %d; remaining refs: %d' % (len(dup_ids), len(kept)))
    if len(dup_ids) > 0:
        with open(new_file, 'w') as f:
            json.dump(kept, f)
        print('saved to ', new_file)


def ref_verify(ref_path='data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup.json'):
    print('ref_verify')
    keys = ['task_id', 'image_id', 'ann_ids', 'Polygons', 'instance_boxes', 'phrase', 'phrase_structure']
    ps_keys = ['name', 'attributes', 'relation_ids', 'relation_descriptions', 'type']
    bad = list()

    with open(ref_path) as f:
        refs = json.load(f)

    for ref in refs:
        good = (len(ref) == len(keys)) and (len(ref['phrase_structure']) == len(ps_keys))
        for k in keys:
            if k not in ref:
                good = False
        for k in ps_keys:
            if k not in ref['phrase_structure']:
                good = False
        if not good:
            bad.append(ref)

    print('bad refs: %d' % len(bad))
    for i, ref in enumerate(bad):
        print(i, ref['task_id'])
        print(list(ref.keys()))
        print(list(ref['phrase_structure'].keys()))
        if i > 10:
            break
    print('bad refs: %d' % len(bad))


def input_only():
    """
    Format of refer_filtered_instance_refine_slim_nodup.json:
    List of dicts ('task_id', 'image_id', 'ann_ids', 'Polygons', 'instance_boxes', 'phrase', 'phrase_structure')
    - phrase_structure: dict of ('name', 'attributes', 'relation_ids', 'relation_descriptions', 'type')

    After input_only:
    refer_filtered_instance_refine_slim_nodup_input.json
    remove gt annotations: Polygons, instance_boxes
    List of dicts ('task_id', 'image_id', 'phrase', 'phrase_structure')
    - phrase_structure: dict of ('name', 'attributes', 'relation_descriptions')
    """
    print('input_only')
    old_file = 'data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup.json'
    new_file = 'data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_input.json'
    with open(old_file) as f:
        refs = json.load(f)

    for task in refs:
        for k in ['Polygons', 'instance_boxes', 'ann_ids']:
            del task[k]
        for k in ['relation_ids', 'type']:
            del task['phrase_structure'][k]

    with open(new_file, 'w') as f:
        json.dump(refs, f)
    print('%s saved.' % new_file)


def no_coco_in_test():
    """
    remove imgs in coco trainval from test split
    image_data_split3000_100_slim.json --> image_data_split3000_100_slim_nococo.json
    refer_filtered_instance_refine_slim_nodup_input_test.json --> refer_filtered_instance_refine_slim_nodup_input_test_nococo.json
    refer_filtered_instance_refine_slim_nodup_test.json --> refer_filtered_instance_refine_slim_nodup_test_nococo.json
    """
    with open('data/refvg/image_data_split3000_100_slim.json') as f:
        info = json.load(f)
    to_remove = set()
    for img in info:
        if img['split'] == 'test' and img['coco_id'] is not None:
            to_remove.add(img['image_id'])
    print('%d imgs to remove from test' % len(to_remove))
    new_info = [img for img in info if img['image_id'] not in to_remove]
    print(len(info), len(new_info))

    # with open('data/refvg/image_data_split3000_100_slim_nococo.json', 'w') as f:
    #     json.dump(new_info, f)
    # print('data/refvg/image_data_split3000_100_slim_nococo.json saved.')

    with open('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_input_test.json') as f:
        ref = json.load(f)
    new_ref = [t for t in ref if t['image_id'] not in to_remove]
    print(len(ref), len(new_ref))
    with open('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_input_test_nococo.json', 'w') as f:
        json.dump(new_ref, f)
    print('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_input_test_nococo.json saved.')

    with open('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_test.json') as f:
        ref = json.load(f)
    new_ref = [t for t in ref if t['image_id'] not in to_remove]
    print(len(ref), len(new_ref))
    with open('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_test_nococo.json', 'w') as f:
        json.dump(new_ref, f)
    print('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_test_nococo.json saved.')


def coco_in_test_to_train():
    """
    move test imgs in coco trainval from test split to train split
    image_data_split3000_100_slim.json --> image_data_split3000_100_slim_nococo.json (overwrite no_coco_in_test())
    refer_filtered_instance_refine_slim_nodup_input_train.json --> refer_filtered_instance_refine_slim_nodup_input_train_nococo.json
    refer_filtered_instance_refine_slim_nodup_train.json --> refer_filtered_instance_refine_slim_nodup_train_nococo.json
    """
    with open('data/refvg/image_data_split3000_100_slim.json') as f:
        info = json.load(f)
    to_move = set()
    for img in info:
        if img['split'] == 'test' and img['coco_id'] is not None:
            to_move.add(img['image_id'])
            img['split'] = 'train'
    print('%d imgs to move from test to train' % len(to_move))

    with open('data/refvg/image_data_split3000_100_slim_nococo.json', 'w') as f:
        json.dump(info, f)
    print(len(info))
    print('data/refvg/image_data_split3000_100_slim_nococo.json saved.')

    with open('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_input_test.json') as f:
        ref = json.load(f)
    new_ref = [t for t in ref if t['image_id'] in to_move]
    with open('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_input_train.json') as f:
        ref = json.load(f)
    print(len(ref))
    ref += new_ref
    print(len(ref))
    with open('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_input_train_nococo.json', 'w') as f:
        json.dump(ref, f)
    print('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_input_train_nococo.json saved.')

    with open('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_test.json') as f:
        ref = json.load(f)
    new_ref = [t for t in ref if t['image_id'] in to_move]
    with open('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_train.json') as f:
        ref = json.load(f)
    print(len(ref))
    ref += new_ref
    print(len(ref))
    with open('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_train_nococo.json', 'w') as f:
        json.dump(ref, f)
    print('data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_train_nococo.json saved.')


if __name__ == '__main__':
    # slim_ref()
    # update_split()
    # split_scene_graph()
    # slim_ref2()
    # remove_ref_duplicate()
    # ref_verify()

    # split_ref()
    # input_only()
    # split_ref(ref_pre='data/refvg/amt_result/refer_filtered_instance_refine_slim_nodup_input')

    no_coco_in_test()
    coco_in_test_to_train()
    # split_scene_graph()
