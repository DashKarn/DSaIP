from transform import FourierTransform
import numpy as np
import matplotlib.pyplot as plt

N = 16


def y_fun(x):
    return np.cos(7*x)


def z_fun(x):
    return np.sin(2*x)

def convolution(sig1, sig2):
    count = 0
    l = len(sig1)
    z = np.zeros(l)
    s = np.copy(sig2)
    s = np.roll(np.flip(s), 1)
    for i in range(l):
        z[i] = np.sum(sig1 * np.roll(s, i))/l
        count += 2*l

    return (z , count)

def correlation(sig1, sig2):
    count = 0
    l = len(sig1)
    z = np.zeros(l)
    for i in range(l):
        z[i] = np.sum(sig1 * np.roll(sig2, i))/l
        count += 2*l
    return (z , count)

def _main():

    t = np.linspace(0, np.pi, N)
    y_values = y_fun(t)
    z_values = z_fun(t)

    correl_values, nr = correlation(z_values, y_values)
    conv_values, nv = convolution(z_values, y_values)

    ty_values = FourierTransform._fft(FourierTransform, y_values, 1)
    tz_values = FourierTransform._fft(FourierTransform, z_values, 1)

    fft_correl_values = FourierTransform.fft_correlation(FourierTransform, ty_values, tz_values)
    fft_conv_values = FourierTransform.fft_convolution(FourierTransform, ty_values, tz_values)

    icorrel_values = FourierTransform._fft(FourierTransform, fft_correl_values, -1)
    iconv_values = FourierTransform._fft(FourierTransform, fft_conv_values, -1)

    print("Convolution + correlation = ", nr+nv)
    print("FFT convolution + correlation = ", FourierTransform.count)

    f, axarr = plt.subplots(3, 2, figsize=(8, 6))
    plt.tight_layout()

    axarr[0, 0].plot(t, y_values)
    axarr[0, 0].set_title('y = cos(7x)')
    axarr[0, 0].grid(True)

    axarr[0, 1].plot(t, z_values)
    axarr[0, 1].set_title('z = sin(2x)')
    axarr[0, 1].grid(True)

    axarr[1, 0].plot(t, list(map(lambda x: x.real, iconv_values)))
    axarr[1, 0].set_title('Результат свертки с БПФ')
    axarr[1, 0].grid(True)

    axarr[1, 1].plot(t, list(map(lambda x: x.real, icorrel_values)))
    axarr[1, 1].set_title('Результат корреляции с БПФ')
    axarr[1, 1].grid(True)

    axarr[2, 0].plot(t, list(map(lambda x: x.real, conv_values)))
    axarr[2, 0].set_title('Результат свертки')
    axarr[2, 0].grid(True)

    axarr[2, 1].plot(t, list(map(lambda x: x.real, correl_values)))
    axarr[2, 1].set_title('Результат корреляции')
    axarr[2, 1].grid(True)

    plt.show()


if __name__ == "__main__":
    _main()
