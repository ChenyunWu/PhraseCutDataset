import numpy as np


class WordEmbed:
    def __init__(self, lookup_path='data/fast_text/lookup_vgpp_tv_rca.npy', vocab_size=-1, word_freq_thresh=1000,
                 init_embed='fast_text'):

        lookup = np.load(lookup_path).item()
        end_ix = len(lookup['ix_to_word'])  # word[end_ix] is excluded from vocab
        if vocab_size > 0:
            end_ix = min(vocab_size, end_ix)
        if word_freq_thresh > 0:
            new_end_ix = end_ix
            for i, f in enumerate(lookup['freq'][:end_ix]):
                if f < word_freq_thresh:
                    new_end_ix = i
                    break
            end_ix = new_end_ix

        self.ix_to_word = lookup['ix_to_word'][:end_ix]
        self.word_to_ix = {word: ix for ix, word in enumerate(self.ix_to_word)}
        self.vocab_size = end_ix

        print('vocabulary size: %d; minimum word frequency: %d' % (end_ix, lookup['freq'][end_ix - 1]))

        if init_embed =='fast_text':
            self.embeddings = lookup['embeddings'][:end_ix]
        elif init_embed == 'random':
            self.embeddings = np.random.randn(end_ix, 300)
        else:
            self.embeddings = None

    def encode_sentences_to_labels(self, sent_str_list, label_length):
        """Input:
        sent_str_list: list of n sents in string format
        return int32 (n, label_length) zeros padded in end
        """
        num_sents = len(sent_str_list)
        labels = np.zeros((num_sents, label_length), dtype=np.int32)
        for i, sent_str in enumerate(sent_str_list):
            if isinstance(sent_str, list):
                tokens = sent_str
            else:
                tokens = sent_str.split()
            for j, w in enumerate(tokens):
                if j < label_length:
                    labels[i, j] = self.word_to_ix[w] if w in self.word_to_ix else self.word_to_ix['<UNK>']
        return labels

    def decode_labels_to_sentences(self, labels):
        """
        labels: int32 (n, label_length) zeros padded in end
        return: list of sents in string format
        """
        # print(labels)
        decoded_sent_strs = []
        num_sents = labels.shape[0]
        for i in range(num_sents):
            label = labels[i].tolist()
            sent_str = ' '.join([self.ix_to_word[int(i)] for i in label if i != 0])
            decoded_sent_strs.append(sent_str)
        return decoded_sent_strs
