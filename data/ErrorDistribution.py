import matplotlib.pyplot as plt
import numpy as np


def error_distribution(error_percentage):

    x = np.array(error_percentage)

    # 直方图会进行统计各个区间的数值
    plt.hist(x, bins=10)

    plt.xlabel('error')
    plt.ylabel('count')

    plt.show()
