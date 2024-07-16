import math
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import Helper
from Second.t6_fc2 import AudioEmotionModel as Model
from torchsummary import summary  # 这个库可以查看用pytorch搭建的模型，各层输出尺寸
import datetime  # 日期时间模块

import os

from utils.Matrix import plot_confusion_matrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

helper = Helper.Helper()
model_name = Model.__name__
writer = SummaryWriter(os.path.join('runs', model_name))

torch.backends.cudnn.enabled = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
num_epochs = 100
decay_epoch = 40
num_classes = 5
batch_size = 64
learning_rate = 0.001
weight_decay = 0.001
momentum = 0.9
print_counter = 5
data_dir = r'/home/AI2022/ycj/1_shuoshi/data/root'


save_dir = os.path.join('result', model_name)


if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 日志文件
sys.stdout = Helper.Logger(os.path.join(save_dir, 'net.txt'.format()))

# 加载 训练集 测试集
trainDataset = Helper.Dataset(os.path.realpath(os.path.join(data_dir, 'train_x.npy')),
                              os.path.realpath(os.path.join(data_dir, 'train_y.npy')))  # 数据集加载
train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True,
                          num_workers=0)  # num_workers为多线程y

testDataset = Helper.Dataset(os.path.realpath(os.path.join(data_dir, 'test_x.npy')),
                             os.path.realpath(os.path.join(data_dir, 'test_y.npy')))
test_loader = DataLoader(testDataset, batch_size, shuffle=True, num_workers=0)


# 固定随机种子，固定初始化
helper.setup_seed(20)
# 初始化模型
model = Model(num_classes=num_classes)
model = model.to(device)


print('时间: ', datetime.datetime.now())
print(f'Batch_Size: {batch_size}, Learning_Rate: {learning_rate}, epochs: {num_epochs}')
summary(model, input_size=(16, 1, 64000))
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 优化算法

# Train the model
total_samples = len(trainDataset)
n_iterations = math.ceil(total_samples / batch_size)  # 向上取整数

tr_number = 0
tr_pre = 0
'''
    model.train(): 启用 BatchNormalization 和 Dropout
    model.eval(): 不启用 BatchNormalization 和 Dropout
'''
# model.train()  # Set model to train mode
early_stop_array = [None] * num_epochs  # 创建相同长度的空数组
counter_time = 0
counter_freq = 0
pic_iterations, pic_loss = [], []
max_acc = []
val_List = []
train_list = []
valid_loss_curve = []
train_loss_curve = []

for epoch in range(num_epochs):
    tr_pre, tr_number = 0, 0
    train_loss_mean = 0
    valid_loss_mean = 0
    helper.adjust_learning_rate(optimizer, epoch, learning_rate, decay_epoch=decay_epoch)  # 学习率动态衰减
    save_va_acc = 0
    for i, (images, labels) in enumerate(train_loader):
        model.train()

        # 最大值归一化
        images = images.reshape(-1, 1, 64000).to(device)
        labels = labels.reshape(-1).to(device)

        # 二次打乱数据
        images, labels = helper.shuffle_Data(images, labels)

        outputs = model(images)
        loss = criterion(outputs, labels.long())

        train_loss_mean += loss.item()

        if i != 0:
            i = i + n_iterations * epoch
            pic_iterations.append(int(i)), pic_loss.append(
                float(loss.item()))  # 批量数 还有损失值存放在pic_iterations, pic_loss = [], []

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, argmax = torch.max(outputs, 1)  # argmax返回分类类别
        tr_number += argmax.size(0)  # tr_number 训练总数
        tr_pre += (labels == argmax).sum().item()  # tr_pre 分类正确数

        iter_number = math.ceil(total_samples / batch_size)
        if num_epochs >= 50:
            print_counter = 1
        valid_step = iter_number // print_counter
        if i % valid_step == (valid_step - 1):  # print every 100 mini-batches

            # #验证集
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.reshape(-1, 1, 64000).to(device)
                    labels = labels.reshape(-1).to(device)

                    out = model(images)
                    loss1 = criterion(out, labels.long())
                    valid_loss_mean += loss1.item()

                    _, predicted = torch.max(out, 1)

                    total += predicted.size(0)
                    correct += (labels == predicted).sum().item()

                va_acc = correct / total
                val = correct / total
                train = tr_pre / tr_number
                val_List.append(float(val))
                train_list.append(float(train))
                max_acc.append(float(va_acc))
                a = loss.item()
                b = loss1.item()
                print(
                    f'epoch: {epoch + 1}/{num_epochs}, step: {i + 1}/{n_iterations}, tr_acc: {tr_pre / tr_number:.4f},va_acc: {va_acc:.4f} ,tr_loss: {a:.4f},vaa_loss:{b:.4f}')

                writer.add_scalar('train_acc', tr_pre / tr_number, epoch)
                writer.add_scalar('val_acc', va_acc, epoch)
                writer.add_scalar('tr_loss', a, epoch)
                writer.add_scalar('vaa_loss', b, epoch)

                if va_acc > save_va_acc:
                    save_va_acc = va_acc
                tr_pre, tr_number = 0, 0
    train_loss_mean /= len(train_loader)
    train_loss_curve.append(train_loss_mean)
    valid_loss_mean /= len(test_loader)
    valid_loss_curve.append(valid_loss_mean)


    def list_max(list):
        index = 0
        max = list[0]
        for i in range(0, len(list)):
            if (list[i] > max):
                max = list[i]
                index = i
        return (index, max)


    list = max_acc
    res = list_max(list)
    print(res)
    torch.save(model.state_dict(), 'result/' + model_name +'/model_' + str(batch_size) + '.pkl')

Emotion_kinds = num_classes
conf_matrix = torch.zeros(Emotion_kinds, Emotion_kinds)


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


# 使用torch.no_grad()可以显著降低测试用例的GPU占用
with torch.no_grad():
    for step, (imgs, targets) in enumerate(test_loader):
        imgs = imgs.reshape(-1, 1, 64000).to(device)
        targets = targets.reshape(-1).to(device)

        out = model(imgs)
        # 记录混淆矩阵参数
        conf_matrix = confusion_matrix(out, targets, conf_matrix)
        conf_matrix = conf_matrix.cpu()

conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
per_kinds = conf_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数

print(conf_matrix)

# 获取每种Emotion的识别准确率
print("每类总个数：", per_kinds)
print("每类预测正确的个数：", corrects)
print("每类的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))

import numpy as np


# 读取实验结果中的精度和损失
def calculate_prediction(metrix):
    """
    计算精度
    """
    label_pre = []
    current_sum = 0
    for i in range(metrix.shape[0]):
        current_sum += metrix[i][i]
        label_total_sum = metrix.sum(axis=0)[i]
        pre = round(100 * metrix[i][i] / label_total_sum, 4)
        label_pre.append(pre)
    print("每类精度：", label_pre)
    all_pre = round(100 * current_sum / metrix.sum(), 4)
    print("总精度：", all_pre)
    return label_pre, all_pre


def calculate_recall(metrix):
    """
    先计算某一个类标的召回率;
    再计算出总体召回率
    """
    label_recall = []
    for i in range(metrix.shape[0]):
        label_total_sum = metrix.sum(axis=1)[i]
        label_correct_sum = metrix[i][i]
        recall = 0
        if label_total_sum != 0:
            recall = round(100 * float(label_correct_sum) / float(label_total_sum), 4)

        label_recall.append(recall)
    print("每类召回率：", label_recall)
    all_recall = round(np.array(label_recall).sum() / metrix.shape[0], 4)
    print("总召回率：", all_recall)
    return label_recall, all_recall


def calculate_f1(prediction, all_pre, recall, all_recall):
    """
    计算f1分数
    """
    all_f1 = []
    for i in range(len(prediction)):
        pre, reca = prediction[i], recall[i]
        f1 = 0
        if (pre + reca) != 0:
            f1 = round(2 * pre * reca / (pre + reca), 4)

        all_f1.append(f1)
    print("每类f1：", all_f1)
    print("总的f1：", round(2 * all_pre * all_recall / (all_pre + all_recall), 4))
    return all_f1


from matplotlib import pyplot as plt

#
loss_x = range(1, len(train_loss_curve) + 1)
train_loss_y = train_loss_curve
valid_loss_y = valid_loss_curve
plt.figure(1)
plt.plot(loss_x, train_loss_y, "r", label="Train_loss")
plt.plot(loss_x, valid_loss_y, "b", label="Valid_loss")
plt.ylabel('loss value')
plt.xlabel('epoch')
plt.legend()
plt.savefig("result/{}/loss_curve.png".format(model_name))

x_label = range(1, len(val_List) + 1)
plt.figure(2)
plt.plot(x_label, train_list, 'r', label='train acc')
plt.plot(x_label, val_List, 'b', label='validation acc')
plt.title('train and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('acc value')
plt.legend()
plt.savefig("result/{}/acc_curve.png".format(model_name))
plt.show()
plt.close()

if __name__ == '__main__':
    metrix = conf_matrix
    print(metrix.sum(axis=0)[0], metrix.sum(axis=1)[0])
    label_pre, all_pre = calculate_prediction(metrix)
    label_recall, all_recall = calculate_recall(metrix)
    # ************************************
    calculate_f1(label_pre, all_pre, label_recall, all_recall)
    trans_mat = np.array(metrix)
    if True:
        labels = ['1', '2']
        label = labels
        plot_confusion_matrix(trans_mat, label, model_name=model_name)


print('时间: ', datetime.datetime.now())

