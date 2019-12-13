
import random
from collections import OrderedDict

from PhraseCutDataset.utils.refvg_loader import RefVGLoader


class VGProcessor(RefVGLoader):

    def __init__(self, split=None, phrase_handler=None, word_embed=None):

        RefVGLoader.__init__(self, split=split, phrase_handler=phrase_handler, word_embed=word_embed,
                             include_vg_scene_graph=True)

        self.phrase_type_stat = {'name': 0, 'attribute': 0, 'relation': 0, 'location': 0, 'verbose': 0}
        self.images = self.vg_loader.images
        self.objects = self.vg_loader.objects
        self.is_same_category = self.vg_loader.is_same_category
        self.is_same_relation = self.vg_loader.is_same_relation

    @staticmethod
    def att_name_phrase(att, name):
        att_words = att.split()
        ph = att + ' ' + name
        if len(att_words) > 1:
            if att_words[0] in ['in', 'on', 'for', 'of', 'with', 'made', 'to', 'not', 'turned', 'off', 'from'] or \
                    (att_words[0][-3:] == 'ing' and att_words[0] not in ['living', 'king', 'ping', 'ceiling']):
                ph = name + ' ' + att
        return ph

    def gen_phrase(self, ref_ann_id, ann_ids):
        ref_ann = self.objects[ref_ann_id]
        if not ref_ann['names']:
            ref_ann['names'] = ['bad_name']
        anns = [self.objects[ann_id] for ann_id in ann_ids]
        unique_type = True
        st_anns = []
        st_atts = []
        st_rels = []
        phrase_structure = {'type': '', 'name': '', 'attributes': [], 'relations': []}
        for ann in anns:
            if ann['ann_id'] == ref_ann['ann_id']:
                continue
            if self.is_same_category(ref_ann, ann):
                unique_type = False
                st_anns.append(ann)
                st_atts += ann['attributes']
                st_rels += ann['relations']

        if unique_type:
            self.phrase_type_stat['name'] += 1
            name = random.choice(ref_ann['names'])
            phrase = name
            phrase_structure['type'] = 'name'
            phrase_structure['name'] = name
            if len(ref_ann['attributes']) > 0:
                att = random.choice(ref_ann['attributes'])
                phrase_structure['attributes'].append(att)
                phrase = self.att_name_phrase(att, phrase)
            elif len(ref_ann['relations']) > 0:
                rels = [r for r in ref_ann['relations'] if len(self.objects[r['object_id']]['names']) > 0]
                if len(rels) > 0:
                    rel = random.choice(rels)
                    phrase_structure['relations'].append(rel)
                    obj_names = self.objects[rel['object_id']]['names']
                    obj_name = random.choice(obj_names)
                    phrase += ' ' + rel['predicate'] + ' ' + obj_name
            return phrase, phrase_structure

        # look for unique attribute
        for att in ref_ann['attributes']:
            if att not in st_atts:
                name = random.choice(ref_ann['names'])
                phrase = self.att_name_phrase(att, name)
                self.phrase_type_stat['attribute'] += 1
                phrase_structure['type'] = 'attribute'
                phrase_structure['name'] = name
                phrase_structure['attributes'].append(att)
                return phrase, phrase_structure

        # look for unique relation
        for ref_rel in ref_ann['relations']:
            for rel in st_rels:
                if not self.is_same_relation(ref_rel, rel):
                    name = random.choice(ref_ann['names'])
                    if len(self.objects[ref_rel['object_id']]['names']) > 0:
                        obj_name = random.choice(self.objects[ref_rel['object_id']]['names'])
                        phrase = name + ' ' + ref_rel['predicate'] + ' ' + obj_name
                        self.phrase_type_stat['relation'] += 1
                        phrase_structure['type'] = 'relation'
                        phrase_structure['name'] = name
                        phrase_structure['relations'].append(rel)  # BUG HERE! should be ref_rel instead of rel
                        return phrase, phrase_structure

        # most verbose phrase
        phrase_str = random.choice(ref_ann['names'])
        phrase_structure['type'] = 'verbose'
        phrase_structure['name'] = phrase_str
        phrase_structure['attributes'] = ref_ann['attributes']
        phrase_structure['relations'] = ref_ann['relations']
        for att in ref_ann['attributes']:
            phrase_str = self.att_name_phrase(att, phrase_str)
        added_rel = []
        for ref_rel in ref_ann['relations']:
            if len(self.objects[ref_rel['object_id']]['names']) > 0:
                obj_name = random.choice(self.objects[ref_rel['object_id']]['names'])
                rel_str = ref_rel['predicate'] + ' ' + obj_name
                if rel_str not in added_rel:
                    phrase_str += ' ' + rel_str
                    added_rel.append(rel_str)
        self.phrase_type_stat['verbose'] += 1
        return phrase_str, phrase_structure

    def recover_relations(self, task, verbose=False):
        """
        Since we only need to recover for type='relation', there should only be one ref-relation.
        """
        assert task['phrase_structure']['type'] == 'relation'
        phrase = task['phrase']
        for ann_id in task['ann_ids']:
            for rel in sorted(self.objects[ann_id]['relations'], key=lambda r: -len(r['predicate'])):
                pred = rel['predicate'].strip()
                pred_space = ' ' + pred  + ' '
                if verbose:
                    print('pred_space:"%s"' % pred_space)
                if pred_space in phrase:
                    so_names = self.objects[rel['object_id']]['names']
                    segs = phrase.split(pred)
                    s = segs[0]
                    for si in range(1, len(segs)):
                        if len(s) > 0 and s[-1] == ' ' and len(segs[si]) > 0 and segs[si][0] == ' ':
                            s += '[SPLIT]' + segs[si]
                        else:
                            s += pred + segs[si]
                    segs = s.split('[SPLIT]')
                    segs = [seg.strip() for seg in segs]
                    if verbose:
                        print('segs', segs)
                        print('so_names', so_names)
                    for seg_i in range(1, len(segs)):
                        n = pred_space.join(segs[seg_i:]).strip()
                        if verbose:
                            print('n:"%s"' % n)
                        if n in so_names:
                            task['phrase_structure']['relations'] = [rel]
                            return

        print('ERR: rel-relation not found')
        print(task['task_id'])
        print('"'+phrase+'"')
        for ann_id in task['ann_ids']:
            for rel in self.objects[ann_id]['relations']:
                print('"'+rel['predicate']+'"', self.objects[rel['object_id']]['names'])
        assert False

    def split_rel_descriptions(self, rds_str, past_rds, predicates, names, strict_skip=True, verbose=False):
        """
        Recursively split a string of several relation descriptions into formatted rel_descriptions
        :param rds_str: unparsed string
        :param past_rds: parsed rel_descriptions: list of (predicate, obj_name) pairs
        :param predicates: list of predicates for remaining relations
        :param names: list of obj_name_list for remaining relations
        :param strict_skip: if true, only allow skipping a rel when it matches with added rel_descriptions
        :param verbose
        :return: Total rel_descriptions: past_rds plus the ones from rds_str; if None: there is no possible parsing
        """
        if verbose:
            print('from split_rel_descriptions: "%s"' % rds_str, past_rds, predicates)
        if len(rds_str) == 0:
            if verbose:
                print('to return: "%s"' % rds_str, past_rds)
            return past_rds
        if len(predicates) == 0:
            return None

        can_skip = False
        for past_pred, past_name in past_rds:
            if past_pred == predicates[0]:
                if past_name in names[0]:
                    can_skip = True
                    break
        if can_skip or not strict_skip:
            skip_rds = self.split_rel_descriptions(rds_str, past_rds, predicates[1:], names[1:], verbose)
            if verbose:
                print('returned skip_rds: ', skip_rds)
            if skip_rds is not None:
                return skip_rds

        if not rds_str.startswith(predicates[0]):
            if verbose:
                print('split_rel_descriptions None-0: ', rds_str, predicates[0])
            return None

        seg = rds_str[len(predicates[0]):].strip()  # remove predicate[0]
        for name in sorted(names[0], key=len, reverse=True):
            if seg.startswith(name):
                new_rds_seg = seg[len(name):].strip()  # remove name
                new_past_rds = past_rds + [(predicates[0], name)]
                rds = self.split_rel_descriptions(new_rds_seg, new_past_rds, predicates[1:], names[1:], verbose)
                if verbose:
                    print('returned rds: ', rds)
                if rds is not None:
                    return rds
        if verbose:
            print('split_rel_descriptions None-1: ', seg, names[0])
        return None

    def gen_rel_descriptions(self, task, verbose=False):
        """
        Recover relation_descriptions in phrase_structure. return it and add it to phrase_structure.
        rel_descriptions are added to the phrase in the same order as in 'relations'.
        A relation could be skipped if it's the same as existing rel_descriptions
        :return: recovered relation_description: list of (predicate, supporting_object_name)
        """
        ps = task['phrase_structure']
        phrase = task['phrase']

        if len(ps['relations']) == 0:
            ps['relation_descriptions'] = list()
            return list()

        word_count_na = len(ps['name'].split())
        for att in ps['attributes']:
            word_count_na += len(att.split())
        ph_words = phrase.split()
        rds_words = ph_words[word_count_na:]
        rds_seg = ' '.join(rds_words)

        past_rds = list()
        predicates = [r['predicate'] for r in ps['relations']]
        names = [self.objects[r['object_id']]['names'] for r in ps['relations']]

        rds = self.split_rel_descriptions(rds_seg, past_rds, predicates, names, strict_skip=True, verbose=verbose)
        if rds is None:
            rds = self.split_rel_descriptions(rds_seg, past_rds, predicates, names, strict_skip=False, verbose=verbose)
        assert rds is not None
        ps['relation_descriptions'] = rds
        return rds

    def recover_name_att_rel_desc(self, task, verbose=False):
        phrase = task['phrase'].lower().strip()
        ps = {'type': '', 'name': '', 'attributes': [], 'relations': [], 'relation_descriptions': []}
        task['phrase_structure'] = ps
        ref_ann_ids = task['ann_ids']
        ref_ann = self.objects[ref_ann_ids[0]]

        # phrase = name
        if phrase in ref_ann['names']:
            ps['name'] = phrase
            return 0

        # phrase = att + name or name + att
        for att in ref_ann['attributes']:
            remain_seg = ''
            if phrase.startswith(att):
                remain_seg = phrase[len(att):].strip()
            elif phrase[-len(att):] == att:
                remain_seg = phrase[:-len(att)].strip()
            if remain_seg in ref_ann['names']:
                ps['attributes'] = [att]
                ps['name'] = remain_seg
                return 1

        # phrase = name + rel
        for name in ref_ann['names']:
            if not phrase.startswith(name):
                continue
            rel_desc = phrase[len(name):].strip()
            # print('name "%s", rd "%s"' %(name, rel_desc))
            for rel in ref_ann['relations']:
                pred = rel['predicate']
                if not rel_desc.startswith(pred):
                    continue
                obj_names = self.objects[rel['object_id']]['names']
                remain_seg = rel_desc[len(pred):].strip()
                # print('pred "%s", rs "%s", names ' %(pred, remain_seg), obj_names)
                if remain_seg in obj_names:
                    ps['name'] = name
                    ps['relations'] = [rel]
                    ps['relation_descriptions'] = [(pred, remain_seg)]
                    return 2

        # phrase = all atts and rels plus one name
        ps['type'] = 'verbose'
        ps['attributes'] = ref_ann['attributes']
        ps['relations'] = ref_ann['relations']
        remain_words = phrase.lower().split()
        for att in ref_ann['attributes']:
            for w in att.split():
                remain_words.remove(w)
        remain_seg = ' '.join(remain_words)
        first_pred = ''
        if len(ref_ann['relations']) > 0:
            first_pred = ' ' + ref_ann['relations'][0]['predicate']
        for name in ref_ann['names']:
            if remain_seg.startswith(name + first_pred):
                ps['name'] = name
                remain_seg = remain_seg[len(name):].strip()
                predicates = [r['predicate'] for r in ref_ann['relations']]
                rel_names = [self.objects[r['object_id']]['names'] for r in ref_ann['relations']]
                rds = self.split_rel_descriptions(remain_seg, [], predicates, rel_names, strict_skip=True,
                                                  verbose=verbose)
                if rds is None:
                    rds = self.split_rel_descriptions(remain_seg, [], predicates, rel_names, strict_skip=False,
                                                      verbose=verbose)
                assert rds is not None
                ps['relation_descriptions'] = rds
                return 3
            if remain_seg == name:  # somehow didn't have the relation in phrase
                ps['name'] = name
                return 3

        print('ERR: cannot find name for type=verbose')
        print(task['task_id'], '"'+task['phrase']+'"', task['phrase_structure'])
        print('"'+remain_seg+'"')
        print(ref_ann['names'])
        assert False

    def recover_phrase_structure(self, task, verbose=False):
        """
        The phrase could be constructed from any ref_ann in task['ann_ids'].
        Recover phrase_structure from the first ref_ann. (same as other data)
        """
        def print_warning(type, case):
            print('WARNING! type=%s, case=%d' % (type, case))
            print(task['task_id'], '"' + task['phrase'] + '"', ps)
            print('ref_ann:', ref_ann)
            print('st_anns:', st_anns)
            return

        warn = verbose
        case = self.recover_name_att_rel_desc(task, verbose)

        ps = task['phrase_structure']
        ref_ann_ids = task['ann_ids']
        ref_ann = self.objects[ref_ann_ids[0]]

        img = self.images[task['image_id']]
        img_ann_ids = img['obj_ids']
        img_anns = [self.objects[ann_id] for ann_id in img_ann_ids]

        unique_type = True
        st_anns = []
        st_atts = []
        st_rels = []

        for ann in img_anns:
            if ann['obj_id'] == ref_ann_ids[0]:
                continue
            if self.is_same_category(ref_ann, ann):
                unique_type = False
                st_anns.append(ann)
                st_atts += ann['attributes']
                st_rels += ann['relations']

        if unique_type:
            ps['type'] = 'name'
            if case == 3 and warn:
                print_warning(ps['type'], case)
            # assert case < 3
            return

        # look for unique attribute
        for att in ref_ann['attributes']:
            if att not in st_atts:
                ps['type'] = 'attribute'
                if case != 1 and warn:
                    print_warning(ps['type'], case)
                # assert case == 1
                return

        # look for unique relation
        for ref_rel in ref_ann['relations']:
            for rel in st_rels:
                if not self.is_same_relation(ref_rel, rel):
                    ps['type'] = 'relation'
                    if case != 2 and warn:
                        print_warning(ps['type'], case)
                    # assert case == 2
                    return

        ps['type'] = 'verbose'
        if not (ps['attributes'] == ref_ann['attributes'] and ps['relations'] == ref_ann['relations']) and warn:
            print_warning(ps['type'], case)
        # assert ps['attributes'] == ref_ann['attributes']
        # assert ps['relations'] == ref_ann['relations']
        return

    def refine_task_data(self, task, verbose=True):
        """
        Remove duplicate ann_ids in the task, and duplicate name/attributes in the phrase_structure.
        Fix the bugs of missing phrase_structure and wrong 'relation' in phrase_structure (when type='relation).
        Add 'relation_descriptions' to the phrase_structure.
        If the phrase needs to be changed, record the old phrase to 'original phrase'
        The task (and its phrase_structure) is modified inplace.
        :param task: dict for a single task, with fields 'phrase', 'phrase_structure', 'image_id', 'ann_ids', ...
        :param verbose:
        :return: a list of which fields are modified, except the 'relation_descriptions'
        """
        if verbose:
            print('start refining task', task['task_id'], task['phrase'], task['phrase_structure'])
        modified = list()
        ps = task['phrase_structure']

        # -- ann_ids --
        ann_ids = list(OrderedDict.fromkeys(task['ann_ids']).keys())
        if len(ann_ids) < len(task['ann_ids']):
            modified.append('ann_ids')
            if verbose:
                print('ann_ids:', task['ann_ids'], ann_ids)
            task['ann_ids'] = ann_ids

        # -- need to recover the whole phrase_structure --
        if len(ps) == 0:
            modified.append('phrase_structure')
            self.recover_phrase_structure(task, verbose)
            ps = task['phrase_structure']
            if verbose:
                print('recover_phrase_structure: ', task['phrase'], ps)
        else:
            # -- need to recover 'relations' --
            if ps['type'] == 'relation':
                modified.append('relation')
                self.recover_relations(task, verbose)
                if verbose:
                    print('recover_relations: ', task['phrase'], ps['relations'])

            # -- add 'relation_descriptions' --
            self.gen_rel_descriptions(task, verbose)
            if verbose and len(ps['relation_descriptions']) > 0:
                print('gen_rel_desc: ', task['phrase'], ps['relation_descriptions'])

        # -- name vs. atts --
        name = ' '.join(ps['name'].split())
        name_words = name.split()
        atts = [' '.join(a.split()) for a in ps['attributes']]
        # remove att == name
        if name in atts:
            atts.remove(name)
        for ai, att in enumerate(atts):
            # trim name
            if name.startswith(att + ' ') and len(name) > len(att):
                name = name[len(att) + 1:]
            # trim att
            if len(att) > len(name) + 1 and att[-len(name)-1:] == ' ' + name:
                atts[ai] = att[:-len(name)-1]
            # last word in att == first word in name
            att_words = att.split()
            if len(att_words) > 1 and att_words[-1] == name_words[0]:
                if att in ['9th street', 'left front', 'waiting for a ball']:
                    continue
                atts[ai] = ' '.join(att_words[:-1])

        # remove duplicated atts
        atts = list(OrderedDict.fromkeys(atts).keys())

        # -- atts vs. rel_descriptions --
        for rel_desc in ps['relation_descriptions']:
            desc = rel_desc[0] + ' ' + rel_desc[1]
            if desc in atts:
                atts.remove(desc)

        # -- refine rel_descriptions --
        keep_ri = list()
        for ri, rel_desc in enumerate(ps['relation_descriptions']):
            # remove duplicates
            if ri == 0 or rel_desc not in ps['relation_descriptions'][:ri]:
                keep_ri.append(ri)
            else:
                continue
            # refine inside the rel_desc
            pred_words = rel_desc[0].split()
            rel_obj_words = rel_desc[1].split()
            # first word in rel-pred == last word in name
            if len(pred_words) > 1 and pred_words[0] == name_words[-1]:
                pred_words = pred_words[1:]
            # last word in rel-pred == first word in rel-obj-name
            if pred_words[-1] == rel_obj_words[0]:
                if len(rel_obj_words) > 1 and rel_obj_words[0] in ['in', 'on', 'of']:
                    rel_obj_words = rel_obj_words[1:]
                elif len(pred_words) > 1:
                    pred_words = pred_words[:-1]
            # duplicate words inside rel-pred
            keep_wi = [0]
            for wi in range(1, len(pred_words)):
                if pred_words[wi] != pred_words[wi - 1]:
                    keep_wi.append(wi)
            pred_words = [pred_words[wi] for wi in keep_wi]
            pred = ' '.join(pred_words)
            rel_obj_name = ' '.join(rel_obj_words)
            ps['relation_descriptions'][ri] = (pred, rel_obj_name)
        # remove duplicates
        rds = [ps['relation_descriptions'][ri] for ri in keep_ri]
        ps['relation_descriptions'] = rds

        # update task
        if name != ps['name']:
            modified.append('name')
            if verbose:
                print('update name: ', ps['name'], name)
            ps['name'] = name
        if atts != ps['attributes']:
            modified.append('attributes')
            if verbose:
                print('update atts: ', ps['attributes'], atts)
            ps['attributes'] = atts

        # -- new phrase --
        ph_str = name
        for att in atts:
            ph_str = self.att_name_phrase(att, ph_str)
        for rel_desc in ps['relation_descriptions']:
            ph_str += ' ' + rel_desc[0] + ' ' + rel_desc[1]
        ph_str = ' '.join(ph_str.split())  # remove redundant space
        if ph_str != task['phrase']:
            modified.append('phrase')
            if verbose:
                print('update phrase: ', task['phrase'], ph_str)
            task['original_phrase'] = task['phrase']
            task['phrase'] = ph_str
        if len(ps['relation_descriptions']) > 0:
            modified.append('relation_descriptions')
        return modified
