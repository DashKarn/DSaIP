import cmath
import math
from scipy.fftpack import fft, ifft


class FourierTransform:
    count = 0

    @staticmethod
    def fft_convolution(self, sig1, sig2):
        self.count += len(sig1)
        return list(map(lambda x, y: x * y, sig1, sig2))

    @staticmethod
    def fft_correlation(self, sig1, sig2):
        self.count += len(sig1)
        return list(map(lambda x, y: x.conjugate() * y, sig1, sig2))

    @staticmethod
    def _fft(self, _in_values, direction):
        _out_values = self._fft_(self, _in_values, direction)

        if direction == -1:
            n = len(_in_values)
            for i in range(n):
                _out_values[i] /= n

        if direction == 1:
            return fft(_in_values)
        else:
            return ifft(_in_values)

    @staticmethod
    def _fft_(self, _values, direction):
        n = len(_values)

        if n == 1:
            return _values

        _values_even = [complex(0, 0)] * (int(n / 2))
        _values_odd = [complex(0, 0)] * (int(n / 2))

        for i in range(n):
            if i % 2 == 0:
                _values_even[int(i / 2)] = _values[i]
            else:
                _values_odd[int(i / 2)] = _values[i]

        _b_even = self._fft_(self, _values_even, direction)
        _b_odd = self._fft_(self, _values_odd, direction)

        w_n = complex(cmath.exp(direction * (-1) * 2 * cmath.pi * complex(0, 1) / n))
        w = 1
        y = [complex(0, 0)] * n

        for i in range(int(n / 2)):
            y[i] = _b_even[i] + _b_odd[i] * w
            y[i + int(n / 2)] = _b_even[i] - _b_odd[i] * w
            w *= w_n
            self.count += 1

        return y

    @staticmethod
    def _fft_reorder(self, _data, _len):
        if _len <= 2:
            return
        for x in range(_len):
            r = self.rev_next(x, _len)
            if r > x:
                temp = _data[x]
                _data[x] = _data[r]
                _data[r] = temp

        return _data

    @staticmethod
    def rev_next(x, n):
        step = math.log2(n)
        r = 0
        while step != 0:
            r <<= 1
            r += (x & 1)
            x >>= 1
            step -= 1

        return r
