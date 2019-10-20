class FileUtils:

    @staticmethod
    def index_sentence(sentence, word_dict):
        """
        Convert sentence to index list
        :param sentence: String
        :param word_dict:
        :return: list
        """
        words = sentence.strip().split()
        return [word_dict.get(word, 0) for word in words]

    @staticmethod
    def divide_sentence(words, seq_size):
        """
        divide sentence & padding
        :param words: document
        :param seq_size: chunk size
        :return: list of chunks
        """
        if len(words) <= seq_size:
            res = [words + [1 for x in range(seq_size - len(words))]]
        else:
            remain = list(words)  # don't operate inplace
            res = list()

            while remain:
                add, remain = remain[:seq_size], remain[seq_size:]
                if len(add) < seq_size // 3:
                    break
                elif len(add) < seq_size:
                    add = add + [1 for x in range(seq_size - len(add))]
                res.append(add)
        return res




