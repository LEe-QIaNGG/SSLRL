import numpy as np
list_a = np.array([1,2,3,4,5])
mask_1 = np.array([True, False, True, False, True])
mask_2 = np.array([True, True, False])
mask_1[mask_1] = mask_2
list_a[mask_1] = np.array([10,20])
print(list_a)
