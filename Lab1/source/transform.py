import cmath


class FourierTransform:
    count = 0

    @staticmethod
    def _function_values(function_, period_, n_):
        interval = period_ / n_
        func_values = []

        x = complex(0, 0)
        for i in range(n_):
            func_values.append(function_(x))
            x += interval

        return func_values

    @staticmethod
    def _dft(self, _values, direction):
        self.count = 0

        N = len(_values)
        _out_values = [complex(0, 0)] * N

        for k in range(N):
            for m in range(N):
                _out_values[k] += _values[m] * cmath.exp(direction * (-1) * cmath.sqrt(-1) * 2 * cmath.pi * k * m / N)
                self.count += 1
        if direction == 1:
            for i in range(N):
                _out_values[i] /= N

        return _out_values


    @staticmethod
    def _fft(self, _in_values, direction):
        self.count = 0
        _out_values = self._fft_(self, _in_values, direction)

        if direction == 1:
            N = len(_in_values)
            for i in range(N):
                _out_values[i] /= N

        return _out_values


    @staticmethod
    def _reverse(_values):
        n = len(_values)

        if n <= 2:
            return _values

        y = [complex(0, 0)] * n

        for i in range(n):
            if i % 2 == 0:
                y[i] = _values[int(i / 2)]
            else:
                y[i] = _values[int(i / 2) + int(n / 2)]

        return y

    def _fft_(self, _values, direction):
        N = len(_values)
        if N == 1:
            return _values

        w_n = complex(cmath.exp(direction * (-2) * cmath.pi * cmath.sqrt(-1) / N))  #инициализация главного корня
        w = 1

        y = [complex(0, 0)] * N

        for i in range(int(N / 2)):
            y[i] = _values[i] + _values[i + int(N / 2)]
            y[i + int(N / 2)] = (_values[i] - _values[i + int(N / 2)]) * w
            w *= w_n
            self.count += 2

        yy = []
        yy += self._fft_(self, y[:int(N / 2)], direction)
        yy += self._fft_(self, y[int(N / 2):], direction)
        yy = self._reverse(yy)
        return yy

