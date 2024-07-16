import sys
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os
from torch.utils.data import Dataset
import torch.nn as nn


class Helper():

    # 混淆矩阵
    def plot_confusion_matrix(self, cm, classes, savename, title='Confusion Matrix'):
        plt.figure(figsize=(12, 8), dpi=100)
        np.set_printoptions(precision=2)

        # 在混淆矩阵中每格的概率值
        ind_array = np.arange(len(classes))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if c > 0.001:
                plt.text(x_val, y_val, "%0.1f" % (c,), color='black', fontsize=15, va='center', ha='center')

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(classes)))
        plt.xticks(xlocations, classes, rotation=90)
        plt.yticks(xlocations, classes)
        plt.ylabel('Actual label')
        plt.xlabel('Predict label')

        # offset the tick
        tick_marks = np.array(range(len(classes))) + 0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        # show confusion matrix
        plt.savefig(savename, format='png')
        plt.show()

    # 设置随机数种子
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    '''
        动态调整学习率，按照给定的epoch数，用指数级下降方式进行学习率衰减
    '''

    def adjust_learning_rate(self, optimizer, epoch, learning_rate, decay_epoch):
        """Sets the learning rate to the initial LR decayed by 10 every decay_epoch epochs"""
        lr = learning_rate * (0.1 ** (epoch // decay_epoch))
        if lr <= 1.0e-6:
            lr = 1.0e-6
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print(param_group['lr'])

    '''
        清空文件夹中的内容
    '''

    def delete_file(self, path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                for f in os.listdir(path_file):
                    path_file2 = os.path.join(path_file, f)
                    if os.path.isfile(path_file2):
                        os.remove(path_file2)

    def shuffle_Data(self, data, labels):

        shuffle = torch.randperm(data.size(0))
        data = data[shuffle]
        labels = labels[shuffle]
        return data, labels

    def same_Probability_deformable(self, outputs, labels):
        labels = labels.cpu().detach().numpy()
        softmax = nn.Softmax(dim=-1)
        outputs = softmax(outputs)
        start = 0
        counter = 0
        number = labels[0]
        temp_predicted = []
        for i in range(len(labels)):

            if (number != labels[i] or i == (len(labels) - 1)):
                number = labels[i]
                end = counter + start

                size = counter
                if (i == len(labels) - 1):
                    size += 1
                    end += 1
                counter = 1

                sub_output = outputs[start: end].cpu().detach().numpy()
                start = end

                dict = {}
                for j in range(10):
                    prob = 0
                    for output in sub_output:
                        prob += output[j]
                    dict[j] = prob

                max = 0
                label = -1
                for key in dict:
                    if dict[key] > max:
                        max = dict[key]
                        label = key

                max_array = np.full(size, label)
                temp_predicted = np.concatenate((temp_predicted, max_array), axis=0)
            else:
                counter += 1

        predicted = torch.from_numpy(temp_predicted.astype(np.int))
        return predicted

    '''
    early_stop 函数
    在验证集中连续五次最高准确率相同时
    退出循环体
    '''

    def early_stop(self, array, epoch, acc):
        array[epoch] = acc

        bool = False
        if (epoch > 6):
            sub_array = array[epoch - 4: epoch + 1]

            sub_acc = sub_array[0]
            for i in sub_array:
                if (i == sub_acc):
                    bool = True
                else:
                    bool = False
                    break

        if bool:
            print('array: ', sub_array)

        return bool, array


# 加载数据集
class Dataset(Dataset):
    def __init__(self, path_x, path_y, label_size=1):
        self.x = torch.from_numpy(np.load(path_x, allow_pickle=True).astype(np.float32))
        self.y = torch.from_numpy(np.load(path_y, allow_pickle=True).reshape(-1, label_size).astype(np.int64))

        print('self.x size', self.x.size())
        print('self.y size', self.y.size())

        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


'''
    保存日志文件
'''


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
