import torch.nn as nn

class LowRankAdapter(nn.Module):
    def __init__(self, chann, rank):
        super().__init__()
        rank = max(1, int(rank))

        self.down = nn.Conv2d(chann, rank, kernel_size=1, bias=True)
        # self.bn = nn.BatchNorm2d(rank, eps=1e-3)
        self.up = nn.Conv2d(rank, chann, kernel_size=1, bias=True)

        # # 可选：把 up 初始化为 0，使 Adapter 初始输出接近 0，
        # # 相当于初始时几乎退化为“共享专家”。如不需要可注释掉。
        # nn.init.zeros_(self.up.weight)
        # nn.init.zeros_(self.up.bias)

    def forward(self, x):
        x = self.down(x)
        # x = self.bn(x)
        # x = F.relu(x)
        x = self.up(x)
        return x