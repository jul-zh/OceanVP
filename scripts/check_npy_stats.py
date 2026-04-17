import numpy as np

path = '/home/yzhidkova/projects/OceanVP/data/ocean/ocean_t0_train_1994_2013_norm.npy'
x = np.load(path)

print("path:", path)
print("shape:", x.shape)
print("dtype:", x.dtype)
print("mean:", float(x.mean()))
print("std:", float(x.std()))
print("min:", float(x.min()))
print("max:", float(x.max()))
