import torch
import torch.nn.functional as F

# 创建一个输入张量
x = torch.randn(1,3, 4, 4)
print(x)
# 对第三和第四个维度进行填充
padded_x = F.pad(x, (0, 0, 0, 0, 1, 1), "constant", 0)
print(padded_x)

