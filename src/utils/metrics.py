"""Metrics.

    Jiaxin Zhuang, lincolnz9511@gmail.com
"""

import numpy as np


def average_precision(correct, predicted):
    """
    mcr is equal to MCA
    :param self:
    :param correct:
    :param predicted:
    :return: acc, mcr, mcp, class_recall, class_precision
    """
    class_num = np.unique(correct).shape[0]
    correct = np.array(correct)
    predicted = np.array(predicted)
    class_precision = []

    for l in range(class_num):
        predicted_l = np.where(predicted == l)[0]
        if len(predicted_l) == 0:
            precison = 0
        else:
            precison = np.sum(correct[predicted_l] == predicted[predicted_l]) \
                / len(predicted_l)
        class_precision.append(precison)
    mcp = np.mean(class_precision)

    return mcp


if __name__ == "__main__":
    correct = [1, 2, 3, 3]
    predict = [2, 2, 3, 3]
    # (0+0.5+0) / 3 = 0.8333
    print(average_precision(correct, predict))
