import pywt
import numpy as np
import matplotlib.pyplot as plt
#signal = [6, 10, 7, 2, 10, 12, 8, 6, 1, 6, 4, 13, 11, 8, 12, 13]
signal = np.sin(np.linspace(0, 2 * np.pi, 16))

analysis_low_filter = [.5, .5]
analysis_high_filter = [.5, -.5]

synthesis_low_filter = [1, 1]
synthesis_high_filter = [1, -1]

filter_bank = [analysis_low_filter, analysis_high_filter, synthesis_low_filter, synthesis_high_filter]
myWavelet = pywt.Wavelet(name="haar")
print(myWavelet.wavefun(1))
print()

def analyse(x):
    high = []
    low = list()
    for i in range(0, len(x)-1, 2):
        high.append((x[i]*analysis_high_filter[0] + x[i+1]*analysis_high_filter[1]))
        low.append((x[i]*analysis_low_filter[0] + x[i+1]*analysis_low_filter[1]))
    print(high)
    print(low)
    print()
    return high, low

def synthese(high, low):
    x = list()
    for i in range(len(high)):
        x.append((low[i]*synthesis_low_filter[0] + high[i]*synthesis_low_filter[1]))
        x.append((low[i]*synthesis_high_filter[0] + high[i]*synthesis_high_filter[1]))
    return x

h1, l1 = analyse(signal)
h2, l2 = analyse(l1)
h3, l3 = analyse(l2)
h4, l4 = analyse(l3)

x4 = synthese(h4, l4)
x3 = synthese(h3, l3)
x2 = synthese(h2, l2)
x1 = synthese(h1, l1)

f, axarr = plt.subplots(5, 2, figsize=(8, 6), sharey=True)
plt.tight_layout()

axarr[0, 0].plot(list(range(16)), signal)
axarr[0, 0].set_title('Начальный сигнал')
axarr[0, 0].grid(True)

axarr[0, 1].plot(list(range(16)), x1)
axarr[0, 1].set_title('Восстановленный сигнал')
axarr[0, 1].grid(True)

axarr[1, 0].plot(list(range(8)), h1)
axarr[1, 0].set_title('Преобразование 1 high')
axarr[1, 0].grid(True)

axarr[1, 1].plot(list(range(8)), l1)
axarr[1, 1].set_title('Преобразование 1 low')
axarr[1, 1].grid(True)

axarr[2, 0].plot(list(range(4)), h2)
axarr[2, 0].set_title('Преобразование 2 high')
axarr[2, 0].grid(True)

axarr[2, 1].plot(list(range(4)),l2)
axarr[2, 1].set_title('Преобразование 2 low')
axarr[2, 1].grid(True)

axarr[3, 0].plot(list(range(2)), h3)
axarr[3, 0].set_title('Преобразование 3 high')
axarr[3, 0].grid(True)

axarr[3, 1].plot(list(range(2)),l3)
axarr[3, 1].set_title('Преобразование 3 low')
axarr[3, 1].grid(True)

axarr[4, 0].plot(list(range(2)), [h4, h4])
axarr[4, 0].set_title('Преобразование 4 high')
axarr[4, 0].grid(True)

axarr[4, 1].plot(list(range(2)),[l4, l4])
axarr[4, 1].set_title('Преобразование 4 low')
axarr[4, 1].grid(True)

plt.show()
