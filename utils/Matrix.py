#coding=utf-8
import matplotlib.pyplot as plt
import numpy
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt




def plot_confusion_matrix(cm, classes, normalize=False, title='State transition matrix', cmap=plt.cm.Blues, model_name='model'):
    plt.figure()
    sns.set(font_scale=2)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if numpy.array(num, dtype=float) > thresh else "black")

    plt.ylabel('Self patt')
    plt.xlabel('Transition patt')

    plt.tight_layout()
    plt.savefig('result/{}/net.png'.format(model_name), transparent=True, dpi=800)



# trans_mat = np.array([( 94,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,
#     0),
# (  0, 117,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,
#     1),
# (  0,   1, 141,   0,   0,   0,   1,   1,   1,   0,   0,   0,   1,   2,
#     0),
# (  0,   0,   0, 161,   0,   0,   0,   1,   0,   1,   1,   1,   0,   0,
#     0),
# (  0,   0,   0,   0, 177,   0,   0,   0,   0,   0,   0,   1,   0,   0,
#     0),
# (  2,   1,   1,   0,   0, 189,   0,   1,   0,   1,   0,   0,   1,   2,
#     1),
# (  0,   0,   0,   0,   2,   1, 188,   0,   0,   0,   0,   0,   0,   2,
#     0),
# (  4,   1,   2,   1,   0,   1,   2, 190,   1,   0,   0,   3,  1,   1,
#     2),
# (  0,   0,   0,   0,   0,   0,   0,   0, 175,   0,   0,  0,   0,   0,
#     0),
# (  0,   2,   0,   0,   0,   2,   1,   0,   2, 175,   0,   2,   0,   0,
#     0),
# (  0,   0,   0,   0,   0,   0,   1,   0,   0,   1, 160,   0,   1,   1,
#     1),
# (  1,   0,   0,   1,   0,   0,   1,   1,   4,   1,   0, 154,   1,   2,
#     0),
# (  0,   0,   0,  0,   0,   1,   1,   0,   0,   0,   0,   0, 102,   2,
#     1),
# (  1,   0,   0,   0,   1,   0,   1,   0,   0,   0,   0,   0,   0, 127,
#     1),
# (  1,   1,   0,   0,   0,   1,   3,   0,   3,   0,   1,   0,   2,   1,
#   147)], dtype=int)
#
# """method 2"""
# if True:
#     labels= ['DL','AG','EC','Ec','CC','PC','NC','TT','PA','AA','ST','Aa','Pc','PP','DM']
#     label = labels
#     plot_confusion_matrix(trans_mat, label)
