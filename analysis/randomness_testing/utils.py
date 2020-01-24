import os
import inspect
import numpy as np

HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
NUMBERS_DIR = os.path.join(HERE_PATH, 'numbers')


def uniform(n = 100):
    return [0 for i in range(n)]

def split(n = 100):
    return [0 if i < n/2 else 1 for i in range(n)]

def alternating(n=100, rate=1):
    string = []
    count = 0
    while len(string) < n:
        if count % 2 == 0:
            for i in range(rate):
                string.append(0)
        elif count % 2 == 1:
            for i in range(rate):
                string.append(1)
        count += 1
    return string[:n]

def pseudo(n=100):
    return list(np.random.randint(0,2, n))

def from_string(string):
    return [int(i) for i in string]

def from_file(filepath):
    with open(filepath, 'r') as f:
        string = f.read()
    string = string.replace(' ', "")
    string = string.replace('\r', "")
    string = string.replace('\n', "")
    string = [int(i) for i in list(string)]
    return string

pi = from_file(os.path.join(NUMBERS_DIR, 'data.pi'))
e = from_file(os.path.join(NUMBERS_DIR, 'data.e'))
sqrt2 = from_file(os.path.join(NUMBERS_DIR, 'data.sqrt2'))
sqrt3 = from_file(os.path.join(NUMBERS_DIR, 'data.sqrt3'))
