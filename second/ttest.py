import time
import torch
import numpy as np

while True:
    t = time.time()
    a = np.zeros((9000000, 2), dtype=np.int64)
    a = a[:900000, :]
    print((time.time() - t) * 1000)