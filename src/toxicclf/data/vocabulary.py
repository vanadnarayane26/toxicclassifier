from collections import Counter

class Vocabulary:
    def __init__(self, max_size: int, min_freq:int):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_freq = Counter()
        
    def __len__(self):
        return len(self.word2idx)
        
    def build_vocab(self, tokenized_text):
        for tokens in tokenized_text:
            self.word_freq.update(tokens)
            
        filtetred_words = {word: freq for word, freq in self.word_freq.items() if freq >= self.min_freq}
        sorted_words = sorted(filtetred_words.items(), key=lambda x: x[1], reverse=True)[:self.max_size - 2]
        
        for idx, (word, freq) in enumerate(sorted_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
    def word_to_index(self, tokens):
        return [
            self.word2idx.get(token, self.word2idx["<UNK>"])
            for token in tokens
        ]
        