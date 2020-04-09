
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


import torch
num_actions = 4

a = torch.randint(0, num_actions, (1,))
print(a)

b = torch.Tensor([[1,2,3],[4,5,6]])
print(b)
index_1 = torch.LongTensor([[0,1],[2,0]])
index_2 = torch.LongTensor([[0,1,1],[0,0,0]])
print(torch.gather(b, dim=1, index=index_1))
print(torch.gather(b, dim=0, index=index_2))


weights = torch.Tensor([0.1, 10, 3, 0.1]) # create a Tensor of weights
print(torch.multinomial(weights, 1)[0])
