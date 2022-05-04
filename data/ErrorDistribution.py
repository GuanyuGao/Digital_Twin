import matplotlib.pyplot as plt
import numpy as np


def error_distribution(error_percentage):

    x = np.array(error_percentage)

    # 直方图会进行统计各个区间的数值
    plt.hist(x, [0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.xlabel('error')
    plt.ylabel('count')

    plt.show()
