import json

from file_paths import name_att_rel_count_fpath


class PhraseHandler(object):

    def __init__(self, word_embed=None, phrase_length=10,
                 cat_count_thresh=21, att_count_thresh=21, rel_count_thresh=21):

        self.word_embed = word_embed
        self.phrase_length = phrase_length

        # load cat/att/rel count data
        print('PhraseHandler loading nar_count: %s' % name_att_rel_count_fpath)
        with open(name_att_rel_count_fpath, 'r') as f:
            count_info = json.load(f)
            self.cat_count_list = count_info['cat']  # list of (cat_name, count), count from high to low
            self.att_count_list = count_info['att']
            self.rel_count_list = count_info['rel']

        # prepare category
        self.cat_to_count = {k: c for (k, c) in self.cat_count_list}
        self.cat_to_count['[INV]'] = 0
        self.cat_to_count['[UNK]'] = 0
        self.label_to_cat = ['[INV]'] + [k for (k, c) in self.cat_count_list if c >= cat_count_thresh] + ['[UNK]']
        self.cat_to_label = {cat: l for l, cat in enumerate(self.label_to_cat)}
        print('Number of categories: %d / %d, frequency thresh: %d (excluding [INV] [UNK])'
              % (len(self.label_to_cat) - 2, len(self.cat_count_list), cat_count_thresh))

        # prepare attributes
        self.att_to_count = {k: c for (k, c) in self.att_count_list}
        self.att_to_count['[INV]'] = 0
        self.att_to_count['[UNK]'] = 0
        self.label_to_att = ['[INV]'] + [k for (k, c) in self.att_count_list if c >= att_count_thresh] + ['[UNK]']
        self.att_to_label = {att: l for l, att in enumerate(self.label_to_att)}
        print('Number of attributes: %d / %d, frequency thresh: %d (excluding [INV] [UNK])'
              % (len(self.label_to_att) - 2, len(self.att_count_list), att_count_thresh))

        # prepare relationships
        self.label_to_rel = ['[INV]'] + [k for (k, c) in self.rel_count_list if c >= rel_count_thresh] + ['[UNK]']
        self.rel_to_label = {rel: l for l, rel in enumerate(self.label_to_rel)}
        print('Number of relationships: %d / %d, frequency thresh: %d (excluding [INV] [UNK])'
              % (len(self.label_to_rel) - 2, len(self.rel_count_list), rel_count_thresh))

    @property
    def vocab_size(self):
        assert self.word_embed is not None
        return len(self.word_embed.ix_to_word)

    def encode_labels(self, sent_str_list):
        assert self.word_embed is not None
        return self.word_embed.encode_sentences_to_labels(sent_str_list, self.phrase_length)

    def decode_labels(self, labels):
        assert self.word_embed is not None
        return self.word_embed.decode_labels_to_sentences(labels)


def construct_phrase(phrase_struct):
    """
    THis func is the same as how the phrases are built in data collection.
    """
    def att_name_phrase(att, name):
        att_words = att.split()
        ph = att + ' ' + name
        if len(att_words) > 1:
            if att_words[0] in ['in', 'on', 'for', 'of', 'with', 'made', 'to', 'not', 'turned', 'off', 'from'] or \
                    (att_words[0][-3:] == 'ing' and att_words[0] not in ['living', 'king', 'ping', 'ceiling']):
                ph = name + ' ' + att
        return ph

    ph_str = phrase_struct['name']
    for att in phrase_struct['attributes']:
        ph_str = att_name_phrase(att, ph_str)
    for rel_desc in phrase_struct['relation_descriptions']:
        ph_str += ' ' + rel_desc[0] + ' ' + rel_desc[1]
    ph_str = ' '.join(ph_str.split())  # remove redundant space
    return ph_str


def construct_phrase_annotation_label(phrase, phrase_structure, label_length):
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
    assert phrase == construct_phrase(phrase_structure)
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
