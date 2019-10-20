from Settings import Settings
import json


class FileUtils:

    @staticmethod
    def save_dict(mapper, file_name):
        """
        Save dictionary to json file
        :param mapper:
        :param file_name:
        :return:
        """
        path = Settings.data_folder + file_name
        with open(path, "w") as file:
            json.dump(mapper, file)
        return

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
        if len(words) <= seq_size // 3:
            return list()
        
        elif len(words) <= seq_size:
            res = [words + [1 for x in range(seq_size-len(words))]]
        else:
            remain = list(words)  # don't operate inplace
            res = list()
            
            while remain:
                add, remain = remain[:seq_size], remain[seq_size:]
                if len(add) < seq_size // 3:
                    break
                elif len(add) < seq_size:
                    add = add + [1 for x in range(seq_size-len(add))]
                res.append(add)
        return res




