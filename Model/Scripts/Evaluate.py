from Settings import Settings
from FileUtils import FileUtils
import numpy as np
import json
import torch


class Evaluator:
    def __init__(self, model, weight_name):
        self.model = model
        self.weight_path = Settings.model_save_folder + weight_name
        with open(Settings.data_folder + "WordToIndex.json", "r") as file:
            self.word_to_index = json.load(file)

        with open(Settings.data_folder + "IndexToLabel.json", "r") as file:
            self.index_to_label = json.load(file)

        # load weights
        self.model.load_state_dict(torch.load(self.weight_path, map_location="cpu"))
        self.model.eval()

    def label_predict(self, sentence):
        """
        In this function, we divide sentence into chunks, and predict label of each chunk;
        To determine the score, we sum the score for each chunk and divide it by num of chunks
        :param sentence: String
        :return: label, score
        """
        index_words = FileUtils.index_sentence(sentence, self.word_to_index)
        chunks = FileUtils.divide_sentence(index_words, Settings.seq_size)
        result = np.zeros(Settings.class_num)
        for chunk in chunks:
            with torch.no_grad():
                chunk = torch.from_numpy(np.asarray(chunk)).view(1, Settings.seq_size)
                predict = self.model(chunk)
                predict = predict.numpy()[0]
                result += predict
        result /= len(chunks)

        target_index = np.argmax(result) + 1
        label = self.index_to_label.get(str(target_index))
        score = np.max(result)
        return label, score

    @staticmethod
    def matrix_tuple(output, target):
        """
        This function is used for generate confusion matrix
        :param output:
        :param target:
        :return:
        """
        f_output = output.cpu() if Settings.cuda else output.clone()
        f_target = target.cpu() if Settings.cuda else target.clone()

        output_res = f_output.detach().numpy()
        target_res = f_target.detach().numpy()
        predicted_index = np.argmax(output_res)
        target_index = np.argmax(target_res)
        result_list = [[int(predicted_index[i]), int(target_index[i])] for i in range(len(predicted_index))]
        return result_list

    def record_matrix(self, use_loader, log_name):
        """
        Use this method for generate logs used for plot confusion matrix
        :param use_loader:
        :param log_name:
        :return:
        """

        data_loader_use = use_loader
        _index = 0
        result = list()
        for _index, data in enumerate(data_loader_use):
            words, target = data['words'], data['label']

            if Settings.cuda:
                words = words.cuda()
                target = target.cuda()

            with torch.no_grad():

                predicted = model(words)
                m_tuple_list = self.matrix_tuple(predicted, target)
                result += m_tuple_list

        print('End of Matrix Record, Save file in {0}'.format(Settings.log_save_folder + log_name))
        print('-' * 99)
        with open(Settings.log_save_folder + log_name, 'w+') as f:
            json.dump(result, f)
        return


if __name__ == '__main__':
    from TextCNN import Model
    model = Model()

    evaluator = Evaluator(model, "modelAfter40.h5")
    ans = evaluator.label_predict("a3b334c6eefd be95012ebf2b 41d67080e078 ff1c26ea0b6f 3397db22bc41 054cc375d1b7 31cbd98f4b3c 4357c81e10c1 e244ebf791b5 b2c878a75d7e bad6ff5dd7bc 6c5aadef4a8c 7485a52d4412 d3be1b0f8fe0 bad6ff5dd7bc f32c03725f3e 4854f270d3c4 7d46278cb412 2e85d01c5d75 26f768da5068 6af770640118 fe3fe35491b4 7145051bef18 a09d2b299608 9d83e581af4b 952661e90000 d7b75f4cda11 bc1402292f3b d3be1b0f8fe0 cf6e30467290 26f768da5068 d90d872dc5d6 c2617a394fdd 6af770640118 eb51798a89e1 9fdfa6d7b021 26f768da5068 6af770640118 758684f862e4 6365c4563bd1 a845e8e2dbe6 6365c4563bd1 fae55d282318 dcd653a3f3be 6b304aabdcee 26f768da5068 3fb046fb884d 564aaf0c408b ba8c02cd03ef 6365c4563bd1 26f768da5068 6af770640118 0c4ce226d9fe 19aa0e73b5e0 f36e139d9400 a0d49113bfc9 6ce6cc5a3203 221d532d377b 26f768da5068 6af770640118 b56c51feefe4 09ba7a326b5f 564aaf0c408b 26f768da5068 6af770640118 6bf9c0cb01b4 e94953618947 c337a85b8ef9 26f768da5068 6af770640118 26f768da5068 6af770640118 26f768da5068 6af770640118 9bc65adc033c 6ce6cc5a3203 40cb5e209b92 10e45001c2f2 fe4e36a54c8c 46c88d9303da 10e45001c2f2 a8aca8858cb9 9bc65adc033c f287c0dfcb7e c33578d25a0d c85b3821556b cb7631b88e51 564aaf0c408b 607e30a9689e 26f768da5068 fadf15c37b09 8b0131ee1005 26f768da5068 84f9f1285b94 07985bc6e17f 72bd4a50cf4a 26f768da5068 6af770640118 6a95ce91efbd f16c17d3d4fe 238b1899dbab 10e45001c2f2 10e45001c2f2 10e45001c2f2 fe3fe35491b4 10e45001c2f2 6365c4563bd1 6bf9c0cb01b4 1cb14c432b35 6af770640118 5948001254b3 26f768da5068 6af770640118 26f768da5068 fa80b6ed74f1 758684f862e4 a54f370ffdf5 29a67ab1c7dc bad6ff5dd7bc 376aa3d8142d 6b304aabdcee 376aa3d8142d 8f7a92cd0ae7 1cb14c432b35 3eee1ce2a7bf 6af770640118 a691ffe843be 687214cd0acb 2b3fda454acc 26f768da5068 aed969aac7a8 6365c4563bd1 eb51798a89e1 6b304aabdcee 20d53168dbb6 26f768da5068 6af770640118 6b304aabdcee 6b304aabdcee eb51798a89e1 6365c4563bd1 46c88d9303da bad6ff5dd7bc 0c4ce226d9fe 97b6014f9e50 10e45001c2f2 10e45001c2f2 c029f0b9bd73 b3e5ec275c6d 87899190d7f9 811481d64823 a65259ff0092 0af03ed987ad 46c88d9303da ec234507458c 10e45001c2f2 e808fbb8f3fc a8ef4039c422 26f768da5068 ff55c7818987 1440cfab16a4 5275e84e47b9 9b88c973ae02 eb51798a89e1 234127a6b69e 10e45001c2f2 1068682ce752 97b6014f9e50 43af6db29054 10e45001c2f2 97b6014f9e50 8f75273e5510 b933270a669d 1946fc9f0277 26f768da5068 6af770640118 10e45001c2f2 6365c4563bd1 a7b399dbd28d 6ce6cc5a3203 10e45001c2f2 4cab260b50f3 6365c4563bd1 9b88c973ae02 abe7d2dd7c9b aa1ef5f5355f f95d0bea231b 10e45001c2f2 aa1ef5f5355f eb51798a89e1 eb51798a89e1 eb51798a89e1 6df520735456 6df520735456 10e45001c2f2 fe3fe35491b4 783c7a00936a 840c0539c0a3 c9b917564931 0b5273ff6b8d 2e85d01c5d75 f002f83b2b85 9689118928dc c85b3821556b 0b5273ff6b8d 9c6d2f61cb39 35fd78ccf2c5 6af770640118 ea05dcbf2b1b 699252d4cf38 abe7d2dd7c9b 26f768da5068 6adb619713dd 8b0fc6d5c7a0 9bc65adc033c 6365c4563bd1 6365c4563bd1 12663854c1ed aa26eb157720 eb51798a89e1 280b508ae263 26f768da5068 84f9f1285b94 5ee06767bc0f 2bcce4e05d9d 7d4501e8b694 8f75273e5510 6b304aabdcee 6ef2ade170d9 d8e5b108e952 c2d5aa22505c 758684f862e4 43af6db29054 a7b399dbd28d 97b6014f9e50 10e45001c2f2 6365c4563bd1 6365c4563bd1 6f7ac9f0e25c 26f768da5068 e149163b21ec 9b39cc95d580 1068682ce752 6ce6cc5a3203 be95012ebf2b 9bc65adc033c 1068682ce752 5948001254b3 53423fdacb45 2e85d01c5d75 72bd4a50cf4a 57e95eefe520 6ce6cc5a3203 90769b70107f b2af4be7c569 6ce6cc5a3203 f3ecb214bd90 97b6014f9e50 564aaf0c408b 10e45001c2f2 10e45001c2f2 e1f00b14b1c5 7c8869aabb1c 6af770640118 1068682ce752 46c88d9303da b68232b0eb63 f95d0bea231b 1068682ce752 29503e65a644 a0d49113bfc9 5474f0279961 29727c063009 0f489cc42bd1 6365c4563bd1 03360e646317 eb51798a89e1 26f768da5068 6af770640118 6ce6cc5a3203 97b6014f9e50 1b6c95839a6d 6b304aabdcee e9e1ca486e15 f95d0bea231b 9db5536263d8 0679dc2df62b c85b3821556b 97b6014f9e50 ddf4525e90e3 a5a83a6a40b7 633ce68d406a abe7d2dd7c9b 3af0c4aa70dd 2d0359e25cc6 bad6ff5dd7bc bc85663b4eca fbb1a70d2795 4bfcec3d7413 1ca3d71cae51 e5b0ce0e1927 88664a49c8d9 19e9f3592995 2784a2673880 56f047e1e190 c337a85b8ef9 a31962fbd5f3 b61f1af56200 036087ac04f9 2bcce4e05d9d 26f768da5068 3718cc54d544 804e45ed9b1a 8e47cf3f63f4 21e314d3afcc 93790ade6682 de9738ee8b24 4357c81e10c1 cc8f6e942cc5 6d25574664d2 9cdf4a63deb0 edbb92428a8d b59e343416f7 1ddb3ca2410f 5b1787f13fd0 1068682ce752 f1a45cc91e7a bb17bb24f83d f92868fed5cf 5e99d31d8fa4 b43b3b2a2808 ad433df6c1b4 dd48aefb3281 98d0d51b397c e764337e7466 2ef7c27a5df4 93c988b67c47 8e47cf3f63f4 21e314d3afcc 46c88d9303da 6bf9c0cb01b4 e94953618947 c337a85b8ef9 e67eb757a353 de9738ee8b24 ec56ff31bb7a cc8f6e942cc5 5db0283c2a5a d70b102993cc 3f2c89b78025 376abf12780a 97b6014f9e50 6b304aabdcee 9a6d33061c3e 6ce6cc5a3203 f95d0bea231b 40154289ad9d e67eb757a353 8e47cf3f63f4 21e314d3afcc 6d25574664d2 6bf9c0cb01b4 801b08c3d068 c337a85b8ef9 9cdf4a63deb0 1357209fd44f b59e343416f7 6f6729c54a07 99642fd98dde 48d657cd9861 422068f04236 b9699ce57810 04d6630c9ebc 59737d64e181 1d3ecce7be96 1ca3f3b4afc8 59737d64e181 2b3336bc6749 1850801b9c05 70a93b7c0a1c 8f783b6383f0 4ebb3fd9652d a1fde4983c10 6688a0504d85 78ba65bc9565 cc429363fb23 e8f705787cc8 e389b91f6450 d38820625542 9e0c01b8b857 0679dc2df62b e332e5196663 897474efb284 b68ca5c9aab8 6bf9c0cb01b4 9bc65adc033c 8f783b6383f0 5e73e7c39cea 04d6630c9ebc 0d857a6347f4 6ca2dd348663 d38820625542 9e0c01b8b857")
    print(ans)

