class SimpleTokenizer:
    def __init__(self, word_index, index_word):
        self.word_index = word_index
        self.index_word = index_word

    def encode(self, text):
        return [self.word_index.get(word, 1) for word in text.split()]  # 1 = <UNK>

    def decode(self, tokens):
        return " ".join([self.index_word.get(i, "<UNK>") for i in tokens])
