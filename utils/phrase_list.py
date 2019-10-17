import numpy as np
import torch


class PhraseList(object):
    """
    Structure that holds a list of phrases as a single tensor.
    """

    def __init__(self, phrases, phrase_structures, vg_loader=None, max_phrase_len=10):
        """
        Arguments:
            phrases (list[str]) list of phrases
            phrase_structures (list[dict]) list of phrase_structures
        """
        self.vg_loader = vg_loader
        self.max_phrase_len = max_phrase_len

        self.phrases = phrases
        self.phrase_structures = phrase_structures

        if vg_loader is not None:
            cat_to_label = vg_loader.cat_to_label
            unk_cat_label = cat_to_label['[UNK]']
            cat_labels = list()
            for pst in phrase_structures:
                cat_labels.append(cat_to_label.get(pst['name'], unk_cat_label))
            self.cat_labels = torch.tensor(np.array(cat_labels))

            att_to_label = vg_loader.att_to_label
            unk_att_label = att_to_label['[UNK]']
            self.att_labels = list()
            # self.att_binary_labels = torch.zeros((len(self.phrases), len(att_to_label)), dtype=torch.int)
            for p_i, pst in enumerate(phrase_structures):
                if 'attributes' not in pst:
                    self.att_labels.append(list())
                    continue
                ph_labels = list()
                for att in pst['attributes']:
                    ph_labels.append(att_to_label.get(att, unk_att_label))
                    # self.att_binary_labels[p_i, label] = 1
                self.att_labels.append(ph_labels)

            if vg_loader.word_embed is not None and max_phrase_len > 0:
                word_embed = vg_loader.word_embed
                self.phrase_word_labels = torch.tensor(word_embed.encode_sentences_to_labels(phrases, max_phrase_len),
                                                       dtype=torch.long)
                cat_names = [pst['name'] for pst in phrase_structures]
                self.cat_word_labels = torch.tensor(word_embed.encode_sentences_to_labels(cat_names, max_phrase_len),
                                                    dtype=torch.long)

    def to(self, *args, **kwargs):
        self.phrase_word_labels = self.phrase_word_labels.to(*args, **kwargs)
        self.cat_word_labels = self.cat_word_labels.to(*args, **kwargs)
        self.cat_labels = self.cat_labels.to(*args, **kwargs)
        # self.att_binary_labels = self.att_binary_labels.to(*args, **kwargs)
        return self

    def __len__(self):
        return len(self.phrases)


def phrase_lists_concat_field(phrase_lists, field):
    if field in ['phrase_word_labels', 'cat_labels', 'cat_word_labels']:  # 'att_binary_labels'
        tensors = [getattr(pl, field) for pl in phrase_lists]
        merged = torch.cat(tensors)
    elif field in ['phrases', 'att_labels']:
        merged = list()
        for pl in phrase_lists:
            merged += getattr(pl, field)
    else:
        raise NotImplementedError
    return merged

    # """
    # Merge multiple phrase_lists into one
    # """
    # phrases, p_structs, vg_loader, max_p_len = list(), list(), None, 0
    # has_word_embed = False
    # for pl in phrase_lists:
    #     phrases += pl.phrases
    #     p_structs += pl.phrase_structures
    #     if vg_loader is None or (not has_word_embed and pl.vg_loader.word_embed is not None):
    #         vg_loader = pl.vg_loader
    #     max_p_len = max(max_p_len, pl.max_phrase_len)
    # return PhraseList(phrases, p_structs, vg_loader, max_p_len)
