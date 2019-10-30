import numpy as np


class WordEmbed:
    def __init__(self, lookup_path='data/fast_text/lookup_refvg_all.npy', vocab_size=-1, word_freq_thresh=0,
                 init_embed='fast_text'):
        lookup = np.load(lookup_path, allow_pickle=True).item()
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

        if init_embed == 'fast_text':
            self.embeddings = lookup['embeddings'][:end_ix]
        elif init_embed == 'random':
            self.embeddings = np.random.randn(end_ix, 300)
        else:
            self.embeddings = None

    def encode_sentences_to_labels(self, sentences, label_length):
        """
        Sentences are encoded with <BOS> at the beginning, <EOS> at the end, <PAD> if less than label_length,
        <UNK> if there are unknown words. label_length includes <BOS> and <EOS>.
        <PAD> --> 0; <UNK> --> 1; <BOS> --> 2; <EOS> --> 3
        input sentences: list of n sentences in string format
        return: int32 (n, label_length) zeros padded in end
        """
        assert self.word_to_ix['<PAD>'] == 0

        num_sents = len(sentences)
        if num_sents == 0:
            return None
        labels = np.zeros((num_sents, label_length), dtype=np.int32)
        for i, sentence in enumerate(sentences):
            words = self.sentence_to_words(sentence)
            for j, w in enumerate(words):
                if j == label_length:
                    break
                labels[i, j] = self.word_to_ix.get(w, self.word_to_ix['<UNK>'])
        return labels

    @staticmethod
    def sentence_to_words(sentence):
        def replace_special(string):
            special = ['-', "'", ',', ':', '<', '.', '/', '?', '*', '"', '\\', '&', '\x00', '`', '!', ']', '[', '+',
                       '@', '(', ')']
            string = string.lower()
            i = 0
            while i < len(string):
                c = string[i]
                if c in special:
                    string = string[:i] + ' ' + c + ' ' + string[i + 1:]
                    i += 2
                i += 1
            return string
        sentence = replace_special(sentence)
        words = sentence.split()
        words = ['<BOS>'] + words + ['<EOS>']
        return words

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
            sent_str = ' '.join([self.ix_to_word[int(i)] for i in label if i != 0 and i != 2 and i != 3])
            decoded_sent_strs.append(sent_str)
        return decoded_sent_strs
