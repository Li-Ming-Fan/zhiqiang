
import torch

"""

#
x = torch.randn(1, requires_grad=True)
y = x ** 3
z = y * y

#
from zhiqiang.utils.torch_utils import GradChecker

grad_checker = GradChecker()
grad_checker.add_to_checker_dict(y, "y")

#
z.backward()

print(grad_checker.get_grad("y"))

for item in grad_checker.grads_dict.keys():
    print(item)
    print(grad_checker.get_grad(item))

"""


## reshape

c = torch.randn((2*5,))
print(c)

d = torch.reshape(c, (2, 5))
print(d)
