import numpy as np
import torch

from phrase_handler import construct_phrase_annotation_label


class PhraseList(object):
    """
    Structure that holds a list of phrases as a single tensor.
    Fields of settings: phrase_handler, max_phrase_len
    Fields of data: phrases, phrase_structures, att_labels, cat_labels, cat_word_labels, phrase_word_labels
    """

    def __init__(self, phrases=None, phrase_structures=None, phrase_handler=None, max_phrase_len=10):
        """
        Arguments:
            phrases (list[str]) list of phrases
            phrase_structures (list[dict]) list of phrase_structures
        """
        self.phrase_handler = phrase_handler
        if self.phrase_handler is not None:
            self.phrase_handler.phrase_length = max_phrase_len

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

        if phrase_handler is not None and phrases is not None:
            unk_cat_label = phrase_handler.cat_to_label['[UNK]']
            unk_att_label = phrase_handler.att_to_label['[UNK]']
            unk_rel_pred_label = phrase_handler.rel_to_label['[UNK]']
            self.cat_labels = list()
            self.att_labels = list()
            self.rel_pred_labels = list()
            self.rel_obj_labels = list()
            for pst in phrase_structures:
                self.cat_labels.append(phrase_handler.cat_to_label.get(pst['name'], unk_cat_label))

                ph_att_labels = list()
                for att in pst.get('attributes', []):
                    ph_att_labels.append(phrase_handler.att_to_label.get(att, unk_att_label))
                self.att_labels.append(ph_att_labels)

                ph_rel_pred_labels = list()
                ph_rel_obj_labels = list()
                for rel_pred, rel_obj in pst.get('relation_descriptions', []):
                    ph_rel_pred_labels.append(phrase_handler.rel_to_label.get(rel_pred, unk_rel_pred_label))
                    ph_rel_obj_labels.append(phrase_handler.cat_to_label.get(rel_obj, unk_cat_label))
                self.rel_pred_labels.append(ph_rel_pred_labels)
                self.rel_obj_labels.append(ph_rel_obj_labels)
            self.cat_labels = torch.tensor(np.array(self.cat_labels))

            if phrase_handler.word_embed is not None and max_phrase_len > 0:
                self.phrase_anno_labels = list()
                for ph, ps in zip(phrases, phrase_structures):
                    anno_label = construct_phrase_annotation_label(ph, ps, max_phrase_len)
                    self.phrase_anno_labels.append(torch.tensor(anno_label, dtype=torch.long))
                self.phrase_anno_labels = torch.stack(self.phrase_anno_labels)

                self.phrase_word_labels = torch.tensor(phrase_handler.encode_labels(phrases), dtype=torch.long)
                cat_names = [pst['name'] for pst in phrase_structures]
                self.cat_word_labels = torch.tensor(phrase_handler.encode_labels(cat_names), dtype=torch.long)

                # one tensor per phrase (with att/rel). the tensor is #att(rel)_in_phrase x max_phrase_len
                self.att_word_labels = list()
                self.rel_pred_word_labels = list()
                self.rel_obj_word_labels = list()
                for pst in phrase_structures:
                    ph_atts = pst.get('attributes', [])
                    if len(ph_atts) > 0:
                        ph_att_word_labels = torch.tensor(phrase_handler.encode_labels(ph_atts), dtype=torch.long)
                        self.att_word_labels.append(ph_att_word_labels)

                    rels = pst.get('relation_descriptions', [])
                    if len(rels) > 0:
                        ph_rel_preds = [r[0] for r in rels]
                        ph_rel_objs = [r[1] for r in rels]
                        ph_rel_pred_word_labels = torch.tensor(phrase_handler.encode_labels(ph_rel_preds),
                                                               dtype=torch.long)
                        ph_rel_obj_word_labels = torch.tensor(phrase_handler.encode_labels(ph_rel_objs),
                                                              dtype=torch.long)
                        self.rel_pred_word_labels.append(ph_rel_pred_word_labels)
                        self.rel_obj_word_labels.append(ph_rel_obj_word_labels)

    def to(self, *args, **kwargs):
        self.phrase_word_labels = self.phrase_word_labels.to(*args, **kwargs)
        self.phrase_anno_labels = self.phrase_anno_labels.to(*args, **kwargs)
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


def concat_phrase_lists(phrase_lists):
    """
    Concatenate PhraseLists to one.
    :param phrase_lists: list of additional PhraseList
    """
    phrase_handler = phrase_lists[0].phrase_handler
    cat_pl = PhraseList(phrase_handler=phrase_handler, max_phrase_len=phrase_handler.phrase_length)
    for field in ['phrases', 'phrase_structures', 'phrase_word_labels', 'phrase_anno_labels',
                  'cat_labels', 'att_labels', 'rel_pred_labels', 'rel_obj_labels',
                  'cat_word_labels', 'att_word_labels', 'rel_pred_word_labels', 'rel_obj_word_labels']:
        cat_pl.__setattr__(field, phrase_lists_concat_field(phrase_lists, field))
    return cat_pl


def phrase_lists_concat_field(phrase_lists, field):
    if field in ['phrase_word_labels', 'phrase_anno_labels', 'cat_labels', 'cat_word_labels']:
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
