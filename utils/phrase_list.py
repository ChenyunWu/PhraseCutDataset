import numpy as np
import torch

from PhraseCutDataset.internal.process.vg_processor import VGProcessor


class PhraseList(object):
    """
    Structure that holds a list of phrases as a single tensor.
    Fields of settings: vg_loader, max_phrase_len
    Fields of data: phrases, phrase_structures, att_labels, cat_labels, cat_word_labels, phrase_word_labels
    """

    def __init__(self, phrases=None, phrase_structures=None, vg_loader=None, max_phrase_len=10):
        """
        Arguments:
            phrases (list[str]) list of phrases
            phrase_structures (list[dict]) list of phrase_structures
        """
        self.vg_loader = vg_loader
        self.max_phrase_len = max_phrase_len

        self.phrases = phrases
        self.phrase_structures = phrase_structures
        self.phrase_word_labels = None
        self.phrase_anno_labels = None

        self.att_labels = None
        self.cat_labels = None
        self.rel_pred_labels = None
        self.rel_obj_labels = None

        self.cat_word_labels = None
        self.att_word_labels = None
        self.rel_pred_word_labels = None
        self.rel_obj_word_labels = None

        if vg_loader is not None and phrases is not None:
            unk_cat_label = vg_loader.cat_to_label['[UNK]']
            unk_att_label = vg_loader.att_to_label['[UNK]']
            unk_rel_pred_label = vg_loader.rel_to_label['[UNK]']
            self.cat_labels = list()
            self.att_labels = list()
            self.rel_pred_labels = list()
            self.rel_obj_labels = list()
            for pst in phrase_structures:
                self.cat_labels.append(vg_loader.cat_to_label.get(pst['name'], unk_cat_label))

                ph_att_labels = list()
                for att in pst.get('attributes', []):
                    ph_att_labels.append(vg_loader.att_to_label.get(att, unk_att_label))
                self.att_labels.append(ph_att_labels)

                ph_rel_pred_labels = list()
                ph_rel_obj_labels = list()
                for rel_pred, rel_obj in pst.get('relation_descriptions', []):
                    ph_rel_pred_labels.append(vg_loader.rel_to_label.get(rel_pred, unk_rel_pred_label))
                    ph_rel_obj_labels.append(vg_loader.cat_to_label.get(rel_obj, unk_cat_label))
                self.rel_pred_labels.append(ph_rel_pred_labels)
                self.rel_obj_labels.append(ph_rel_obj_labels)
            self.cat_labels = torch.tensor(np.array(self.cat_labels))

            if vg_loader.word_embed is not None and max_phrase_len > 0:

                self.phrase_anno_labels = list()
                for ph, ps in zip(phrases, phrase_structures):
                    anno_label = _construct_phrase_annotation_label(ph, ps, max_phrase_len)
                    self.phrase_anno_labels.append(torch.tensor(anno_label,dtype=torch.long))
                self.phrase_anno_labels = torch.stack(self.phrase_anno_labels)

                word_embed = vg_loader.word_embed
                self.phrase_word_labels = torch.tensor(word_embed.encode_sentences_to_labels(phrases, max_phrase_len),
                                                       dtype=torch.long)
                cat_names = [pst['name'] for pst in phrase_structures]
                self.cat_word_labels = torch.tensor(word_embed.encode_sentences_to_labels(cat_names, max_phrase_len),
                                                    dtype=torch.long)
                self.att_word_labels = list()
                self.rel_pred_word_labels = list()
                self.rel_obj_word_labels = list()
                for pst in phrase_structures:
                    ph_atts = pst.get('attributes', [])
                    if len(ph_atts) > 0:
                        ph_att_word_labels = torch.tensor(
                            word_embed.encode_sentences_to_labels(ph_atts, max_phrase_len), dtype=torch.long)
                        self.att_word_labels.append(ph_att_word_labels)

                    rels = pst.get('relation_descriptions', [])
                    if len(rels) > 0:
                        ph_rel_preds = [r[0] for r in rels]
                        ph_rel_objs = [r[1] for r in rels]
                        ph_rel_pred_word_labels = torch.tensor(
                            word_embed.encode_sentences_to_labels(ph_rel_preds, max_phrase_len), dtype=torch.long)
                        ph_rel_obj_word_labels = torch.tensor(
                            word_embed.encode_sentences_to_labels(ph_rel_objs, max_phrase_len), dtype=torch.long)
                        self.rel_pred_word_labels.append(ph_rel_pred_word_labels)
                        self.rel_obj_word_labels.append(ph_rel_obj_word_labels)

    def to(self, *args, **kwargs):
        self.phrase_word_labels = self.phrase_word_labels.to(*args, **kwargs)
        self.cat_word_labels = self.cat_word_labels.to(*args, **kwargs)
        self.cat_labels = self.cat_labels.to(*args, **kwargs)
        for field in ['att_word_labels', 'rel_pred_word_labels', 'rel_obj_word_labels']:
            fl = getattr(self, field)
            for ei, e in enumerate(fl):
                fl[ei] = e.to(*args, **kwargs)
            setattr(self, field, fl)
        return self

    def __len__(self):
        return len(self.phrases)


def _construct_phrase(phrase_struct):
    """
    THis func is the same as how the phrases are built in data collection.
    """
    ph_str = phrase_struct['name']
    for att in phrase_struct['attributes']:
        ph_str = VGProcessor.att_name_phrase(att, ph_str)
    for rel_desc in phrase_struct['relation_descriptions']:
        ph_str += ' ' + rel_desc[0] + ' ' + rel_desc[1]
    ph_str = ' '.join(ph_str.split())  # remove redundant space
    return ph_str


def _construct_phrase_annotation_label(phrase, phrase_structure, label_length):
    """
    phrase is encoded with <BOS> at the beginning, <EOS> at the end, <PAD> if less than label_length,
    <UNK> if there are unknown words. label_length includes <BOS> and <EOS>.
    <PAD> --> 0; <BOS> --> 2; <EOS> --> 3;
    cat: 4, last word for cat: 5
    att: 6, last word for att: 7
    rel-pred: 8, last word for rel-pred: 9
    rel-obj: 10, last word for rel-obj: 11
    return: list of int (len=label_length), zeros padded in end
    """
    assert phrase == _construct_phrase(phrase_structure)
    # name
    anno_labels = [4] * len(phrase_structure['name'].split())
    anno_labels[-1] += 1
    # atts
    for att in phrase_structure['attributes']:
        att_words = att.split()
        att_labels = [6] * len(att_words)
        att_labels[-1] += 1
        if len(att_words) > 1:
            if att_words[0] in ['in', 'on', 'for', 'of', 'with', 'made', 'to', 'not', 'turned', 'off', 'from'] or \
                    (att_words[0][-3:] == 'ing' and att_words[0] not in ['living', 'king', 'ping', 'ceiling']):
                anno_labels += att_labels
            else:
                anno_labels = att_labels + anno_labels
    # rels
    for r_pred, r_obj in phrase_structure['relation_descriptions']:
        r_pred_labels = [8] * len(r_pred.split())
        r_pred_labels[-1] += 1
        anno_labels += r_pred_labels
        r_obj_labels = [10] * len(r_obj.split())
        r_obj_labels[-1] += 1
        anno_labels += r_obj_labels
    # final
    anno_labels = [2] + anno_labels + [3]
    if len(anno_labels) >= label_length:
        anno_labels = anno_labels[:label_length]
    else:
        anno_labels += [0] * (label_length - len(anno_labels))
    return anno_labels


def concat_phrase_lists(phrase_lists):
    """
    Concatenate PhraseLists to one.
    :param phrase_lists: list of additional PhraseList
    """
    cat_pl = PhraseList(vg_loader=phrase_lists[0].vg_loader, max_phrase_len=phrase_lists[0].max_phrase_len)
    for field in ['phrases', 'phrase_structures', 'phrase_word_labels',
                  'cat_labels', 'att_labels', 'rel_pred_labels', 'rel_obj_labels',
                  'cat_word_labels', 'att_word_labels', 'rel_pred_word_labels', 'rel_obj_word_labels']:
        cat_pl.__setattr__(field, phrase_lists_concat_field(phrase_lists, field))
    return cat_pl


def phrase_lists_concat_field(phrase_lists, field):
    if field in ['phrase_word_labels', 'cat_labels', 'cat_word_labels']:
        tensors = [getattr(pl, field) for pl in phrase_lists]
        merged = torch.cat(tensors)
    elif field in ['phrases', 'phrase_structures', 'att_labels', 'rel_pred_labels', 'rel_obj_labels',
                   'att_word_labels', 'rel_pred_word_labels', 'rel_obj_word_labels']:
        merged = list()
        for pl in phrase_lists:
            merged += getattr(pl, field)
    else:
        raise NotImplementedError
    return merged
