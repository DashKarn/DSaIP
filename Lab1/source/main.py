from transform import FourierTransform
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import cmath

N = 64


def _fun(x):
    y = np.cos(7*x) + np.sin(2*x)
    return y


def _test(_values):

    xx = np.arange(N)
    yf = scipy.fftpack.fft(_values)
    pol = list(map(lambda a: cmath.polar(a), yf))  # преобразование к полярным координатам

    ph_val = []   # угол
    amp_val = []  # радиус

    for i in range(len(_values)):
        amp_val.append(pol[i][0])
        ph_val.append(pol[i][1])

    amp, ph = plt.subplots(2, figsize=(8, 6))
    ph[0].plot(xx, amp_val)
    ph[0].set_title('Амплитудный спектр')
    ph[0].grid(True)

    ph[1].plot(xx, ph_val)
    ph[1].set_title('Фазовый спектр')
    ph[1].grid(True)
    plt.show()


def discrete_transform(_values):
    f, axarr = plt.subplots(2, 2, figsize=(8, 6))
    plt.tight_layout()

    x = np.linspace(0.0, np.pi/4, N)
    xx = np.arange(N)
    dft_values = FourierTransform._dft(FourierTransform, _values, 1)  # получает сумму всех гармоник
    idft_values = FourierTransform._dft(FourierTransform, dft_values, -1)  # обратное преобразование

    pol = list(map(lambda x: cmath.polar(x), dft_values))  # преобразование к полярным координатам

    ph_val = []   # угол
    amp_val = []  # радиус

    for i in range(len(dft_values)):
        amp_val.append(pol[i][0])
        ph_val.append(pol[i][1])

    axarr[0, 0].plot(x, _values)
    axarr[0, 0].set_title('Исходная функция')
    axarr[0, 0].grid(True)

    axarr[0, 1].plot(xx, amp_val)
    axarr[0, 1].set_title('Амплитудный спектр')
    axarr[0, 1].grid(True)

    axarr[1, 0].plot(xx, ph_val)
    axarr[1, 0].set_title('Фазовый спектр')
    axarr[1, 0].grid(True)

    axarr[1, 1].plot(x, idft_values)
    axarr[1, 1].set_title('Обратное преобразование')
    axarr[1, 1].grid(True)

    plt.show()


def fast_transform(_values):
    f, axarr = plt.subplots(2, 2, figsize=(8, 6))
    plt.tight_layout()

    xx = np.arange(N)
    x = np.linspace(0.0, np.pi / 4, N)
    fft_values = FourierTransform._fft(FourierTransform, _values, 1)
    ifft_values = FourierTransform._fft(FourierTransform, fft_values, -1)

    pol = list(map(lambda x: cmath.polar(x), fft_values))

    ph_val = []
    amp_val = []

    for i in range(len(fft_values)):
        amp_val.append(pol[i][0])
        ph_val.append(pol[i][1])

    axarr[0, 0].plot(x, _values)
    axarr[0, 0].set_title('Исходная функция')
    axarr[0, 0].grid(True)

    axarr[0, 1].plot(xx, amp_val)
    axarr[0, 1].set_title('Амплитудный спектр')
    axarr[0, 1].grid(True)

    axarr[1, 0].plot(xx, ph_val)
    axarr[1, 0].set_title('Фазовый спектр')
    axarr[1, 0].grid(True)

    axarr[1, 1].plot(x, ifft_values)
    axarr[1, 1].set_title('Обратное преобразование')
    axarr[1, 1].grid(True)

    plt.show()


def _main():
    fun_values = FourierTransform._function_values(_fun, 2 * np.pi, N)  # массив значений у с шагом 2*np.pi/N
    discrete_transform(fun_values)
    print('DFF number of operations:', FourierTransform.count)
    fast_transform(fun_values)
    print('FFT number of operations:', FourierTransform.count)
    _test(fun_values)


_main()
