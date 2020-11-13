import os
import sys

# Adding project folder
root = os.getcwd().replace("\\", "/")
sys.path.append(root)

import tenv.config as config
import tenv.util as util
import pandas as pd
import tenv.network as nw
import numpy as np

from collections import defaultdict

file = open("myfilea.dat", "a")
# for o in range(1000):
#     file.write(np.arange(1000))

# file.close()

# array = np.arange(1000)
# array.astype("int16").tofile("array.dat")

# a = np.load("array.dat")
# print(a)

# file = open("myfile.dat", "rb")
# print(file.read())

a = 256
i = 0
for o in sorted(util.G.nodes()):
    for d in sorted(util.G.nodes()):
        if i >= a:
            break

        i = i + 1
        sp = [o, d] + nw.get_sp(util.G, o, d)
        sp = np.array(sp, dtype=np.int16)
        file.write(str(sp) + "\n")

        print(sp)

file.close()
