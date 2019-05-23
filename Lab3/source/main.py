from source.transform import *
import numpy as np
import matplotlib.pyplot as plt


N = 16

def _fun(x):
    y = np.cos(2*x) + np.sin(7*x)
    return y


def fast_transform():

    x = np.linspace(0.0, 2 * np.pi, N)
    function_values = _fun(x)

    transform = ffut(function_values, N)
    reverse_transform = iffut(transform, N)


    plt.subplot(221)
    plt.plot(x, function_values)
    plt.title('Исходная функция')
    plt.grid(True)

    plt.subplot(222)
    plt.plot([i for i in range(N)], transform)
    plt.title('БПУ')
    plt.grid(True)

    plt.subplot(223)
    plt.plot(x, reverse_transform)
    plt.title('Обратное БПУ')
    plt.grid(True)

    plt.show()


def discrete_transform():
    x = np.linspace(0.0, 2 * np.pi, N)
    function_values = _fun(x)

    transform = ddut(function_values, N)
    reverse_transform = iddut(transform, N)

    plt.subplot(221)
    plt.plot(x, function_values)
    plt.title('Исходная функция')
    plt.grid(True)

    plt.subplot(222)
    plt.plot([i for i in range(N)], transform)
    plt.title('ДПУ')
    plt.grid(True)

    plt.subplot(223)
    plt.plot(x, reverse_transform)
    plt.title('Обратное ДПУ')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    discrete_transform()
    fast_transform()
