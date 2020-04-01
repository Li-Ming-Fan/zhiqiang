
import numpy as np


list_a = ["a", "b", "c", "d", "e"]
list_p = [0.1, 0.1, 0.3, 0.4, 0.1]   # ValueError: probabilities do not sum to 1

sample_shape = (10, 3)

# replace=True，表示可重复采样，False表示不可重复采到同一个元素
a = np.random.choice(list_a, sample_shape, replace=True, p=list_p)
print(a)

#
num_sample = 3
a = np.random.choice(list_a, 3, replace=False, p=list_p)
print(a)

