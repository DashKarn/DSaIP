import numpy as nm
from scipy import linalg

def ffut(signal, N):
    def fun(a):
        l = len(a)
        if l == 1:
            return a
        else:
            up = fun(a[0:int(l / 2)])
            down = fun(a[int(l / 2):l])
            return list(map(lambda x, y: x + y, up, down)) + list(map(lambda x, y: x - y, up, down))

    return list(map(lambda x: x / N, fun(signal)))


def iffut(signal, N):
    def fun(a):
        l = len(a)
        if l == 1:
            return a
        else:
            up = fun(a[0:int(l / 2)])
            down = fun(a[int(l / 2):l])
            return list(map(lambda x, y: x + y, up, down)) + list(map(lambda x, y: x - y, up, down))

    return fun(signal)

def ddut(signal, N):

    walsh = linalg.hadamard(N)

    return nm.dot(walsh, signal)/N


def iddut(signal, N):

    walsh = linalg.inv(linalg.hadamard(N))

    return nm.dot(signal, walsh)*N
