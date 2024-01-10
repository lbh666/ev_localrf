import numpy as np

arr1 = np.zeros((2,3))
arr2 = None
arr3 = np.zeros((2,3))

# 处理 arr2 为 None 的情况
if arr2 is None:
    arr2 = np.array([])

result = np.concatenate((arr1, arr2, arr3))
print(result)