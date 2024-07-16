import librosa
import numpy as np
import os
from tqdm import *
from sklearn.model_selection import train_test_split
from ssqueezepy import cwt, stft, ssq_cwt, ssq_stft, Wavelet
import soundfile as sf
from scipy.signal import resample
from ssqueezepy import ssq_stft


# 获取读取文件路径
def get_wav_files(parent_dir, sub_dirs):
    wav_files = []
    for l, sub_dir in enumerate(sub_dirs):
        wav_path = os.path.join(parent_dir, sub_dir)
        for (dirpath, dirnames, filenames) in os.walk(wav_path):  # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
            for filename in filenames:
                if filename.endswith('.wav') or filename.endswith('.WAV'):
                    filename_path = os.sep.join([dirpath, filename])
                    filename_path = os.path.realpath(filename_path)
                    wav_files.append(filename_path)
    return wav_files


parent_dir = r'/home/AI2022/ycj/1_shuoshi/data/wav'
data_save_root = r'/home/AI2022/ycj/1_shuoshi/data/dog_root_envelope'  # 数据保存地址
sub_dir = ['00', '01', '02', '03', '04']  # 子目录
# sub_dir = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10']
file_names = get_wav_files(parent_dir, sub_dir)  # 获取数据文件夹下的所有文件路径

if not os.path.exists(data_save_root):  # 判断是否存在保存文件的文件夹
    os.makedirs(data_save_root)

features_all = []
labels_all = []
index = 0

def feature_out(y_16k, label):
    ssq_spec, _, _, _ = ssq_stft(x=y_16k, n_fft=512, hop_len=160, window='hann', fs=16000)
    features_all.append(ssq_spec)
    labels_all.append(label)
    return 0


for file in tqdm(file_names):
    label_str = file.split('/')[-2]
    label = int(label_str)
    y, sr = librosa.load(file, sr=16000)  # 使用原始采样频率读取数据文件

    if label_str == '00' or label_str == '01':
        #  long
        if len(y) >= 64000:
            y1 = y[:64000]
            feature_out(y1, label)

        else:
            number = 64000 - int(len(y))
            array = np.full(number, 0)
            y2 = np.concatenate((y, array), axis=0)
            feature_out(y2, label)

    elif label_str == '02' or label_str == '03' or label_str == '04':
        # short
        if len(y) >= 80000:
            y1 = y[:64000]
            y2 = y[len(y) - 64000:len(y)]
            feature_out(y1, label)
            feature_out(y2, label)


        elif len(y) < 80000 and len(y) >= 64000:
            y3 = y[:64000]
            feature = feature_out(y3, label)
        elif len(y) < 64000:
            number = 64000 - int(len(y))
            array = np.full(number, 0)
            y4 = np.concatenate((y, array), axis=0)
            feature_out(y4, label)

    else:
        print("数据读取一次还是两次分类错误！")

print('features_all:{}'.format(np.shape(features_all)))
print('labels_all:{}'.format(np.shape(labels_all)))

images = np.array(features_all, dtype=float)
labels = np.array(labels_all, dtype=int)

test_x, train_x, test_y, train_y = train_test_split(images, labels, train_size=0.2, stratify=labels, shuffle=True)
np.save(data_save_root + "/train_x.npy", train_x)
np.save(data_save_root + "/train_y.npy", train_y)
np.save(data_save_root + "/test_x.npy", test_x)
np.save(data_save_root + "/test_y.npy", test_y)
