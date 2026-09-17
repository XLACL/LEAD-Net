import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DLRA import LowRankAdapter

# =============================================================================
# 1. Downsampler
# =============================================================================
class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, nb_tasks=1):
        super().__init__()

        self.conv = nn.Conv2d(
            ninput, noutput - ninput, (3, 3),
            stride=2, padding=1, bias=True,
        )
        self.pool = nn.MaxPool2d(2, stride=2)

        self.bn_ini = nn.ModuleList([
            nn.BatchNorm2d(noutput, eps=1e-3)
            for _ in range(nb_tasks)
        ])

    def forward(self, input, task):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn_ini[task](output)
        return F.relu(output)


# =============================================================================
# 2. Upsampler
# =============================================================================
class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput, nb_tasks=1):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3,
            stride=2, padding=1, output_padding=1, bias=True,
        )
        self.bn_up = nn.ModuleList([
            nn.BatchNorm2d(noutput, eps=1e-3)
            for _ in range(nb_tasks)
        ])

    def forward(self, input, task):
        output = self.conv(input)
        output = self.bn_up[task](output)
        return F.relu(output)

# =============================================================================
# 4. RAP Block：Shared Expert + Domain-Specific Expert (Low-Rank Adapter)
#
#   每个 sub-stage:
#       fused = shared(x) + adapter[task](x)
#       out   = ReLU(BN[task](fused))
#
#   两个 sub-stage 结构相同，第二个带 dilation。
# =============================================================================
class non_bottleneck_1d_RAP(nn.Module):
    def __init__(self, chann, dropprob, dilated, nb_tasks=1, adapter_rank=8):
        super().__init__()

        self.nb_tasks = nb_tasks
        self.chann = chann

        # ---- Shared Expert - sub-stage 1 ----
        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1),
            stride=1, padding=(1, 0), bias=True,
        )
        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3),
            stride=1, padding=(0, 1), bias=True,
        )

        # ---- Domain-Specific Expert (Low-Rank Adapter) - sub-stage 1 ----
        self.adapters_1 = nn.ModuleList([
            LowRankAdapter(chann, adapter_rank)
            for _ in range(nb_tasks)
        ])
        self.bns_1 = nn.ModuleList([
            nn.BatchNorm2d(chann, eps=1e-3)
            for _ in range(nb_tasks)
        ])

        # ---- Shared Expert - sub-stage 2 ----
        self.conv3x1_2 = nn.Conv2d(
            chann, chann, (3, 1),
            stride=1, padding=(1 * dilated, 0), bias=True,
            dilation=(dilated, 1),
        )
        self.conv1x3_2 = nn.Conv2d(
            chann, chann, (1, 3),
            stride=1, padding=(0, 1 * dilated), bias=True,
            dilation=(1, dilated),
        )

        # ---- Domain-Specific Expert (Low-Rank Adapter) - sub-stage 2 ----
        self.adapters_2 = nn.ModuleList([
            LowRankAdapter(chann, adapter_rank)
            for _ in range(nb_tasks)
        ])
        self.bns_2 = nn.ModuleList([
            nn.BatchNorm2d(chann, eps=1e-3)
            for _ in range(nb_tasks)
        ])

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input, task):
        if task >= self.nb_tasks:
            raise ValueError(
                f"任务ID {task} 超出范围 (0-{self.nb_tasks - 1})"
            )

        # =====================================================================
        # sub-stage 1
        # =====================================================================
        shared_1 = self.conv3x1_1(input)
        shared_1 = F.relu(shared_1)
        shared_1 = self.conv1x3_1(shared_1)

        current_1 = self.adapters_1[task](input)

        fused_1 = shared_1 + current_1
        out_1 = self.bns_1[task](fused_1)
        out_1 = F.relu(out_1)

        # =====================================================================
        # sub-stage 2
        # =====================================================================
        shared_2 = self.conv3x1_2(out_1)
        shared_2 = F.relu(shared_2)
        shared_2 = self.conv1x3_2(shared_2)

        current_2 = self.adapters_2[task](out_1)

        fused_2 = shared_2 + current_2
        out_2 = self.bns_2[task](fused_2)

        if self.dropout.p != 0:
            out_2 = self.dropout(out_2)

        return F.relu(out_2 + input)


# =============================================================================
# 5. Encoder
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, nb_tasks=1, adapter_rank=8):
        super().__init__()

        self.initial_block = DownsamplerBlock(3, 16, nb_tasks)

        self.layers = nn.ModuleList()
        self.layers.append(DownsamplerBlock(16, 64, nb_tasks))

        for _ in range(5):
            self.layers.append(
                non_bottleneck_1d_RAP(64, 0.03, 1, nb_tasks, adapter_rank)
            )

        self.layers.append(DownsamplerBlock(64, 128, nb_tasks))

        for _ in range(2):
            self.layers.append(
                non_bottleneck_1d_RAP(128, 0.3, 2, nb_tasks, adapter_rank)
            )
            self.layers.append(
                non_bottleneck_1d_RAP(128, 0.3, 4, nb_tasks, adapter_rank)
            )
            self.layers.append(
                non_bottleneck_1d_RAP(128, 0.3, 8, nb_tasks, adapter_rank)
            )
            self.layers.append(
                non_bottleneck_1d_RAP(128, 0.3, 16, nb_tasks, adapter_rank)
            )

    def forward(self, input, task):
        output = self.initial_block(input, task=task)
        for layer in self.layers:
            output = layer(output, task=task)
        return output


# =============================================================================
# 6. Decoder
# =============================================================================
class Decoder(nn.Module):
    def __init__(self, num_classes, nb_tasks, adapter_rank=8):
        super().__init__()

        self.layers = nn.ModuleList([
            UpsamplerBlock(128, 64, nb_tasks),
            non_bottleneck_1d_RAP(64, 0, 1, nb_tasks, adapter_rank),
            non_bottleneck_1d_RAP(64, 0, 1, nb_tasks, adapter_rank),
            UpsamplerBlock(64, 16, nb_tasks),
            non_bottleneck_1d_RAP(16, 0, 1, nb_tasks, adapter_rank),
            non_bottleneck_1d_RAP(16, 0, 1, nb_tasks, adapter_rank),
        ])

        self.output_conv = nn.ModuleList([
            nn.ConvTranspose2d(
                16, num_classes[i], 2,
                stride=2, padding=0, output_padding=0, bias=True,
            )
            for i in range(nb_tasks)
        ])

    def forward(self, input, task):
        output = input
        for layer in self.layers:
            output = layer(output, task=task)
        output = self.output_conv[task](output)
        return output


# =============================================================================
# 7. Network
# =============================================================================
class Net(nn.Module):
    def __init__(self, num_classes=[20], nb_tasks=1, adapter_rank=8):
        super().__init__()
        self.encoder = Encoder(nb_tasks, adapter_rank=adapter_rank)
        self.decoder = Decoder(num_classes, nb_tasks, adapter_rank=adapter_rank)

    def forward(self, input, task):
        output = self.encoder(input, task=task)
        output = self.decoder(output, task=task)
        return output


# =============================================================================
# 8. 使用示例
# =============================================================================
if __name__ == "__main__":
    nb_tasks = 3
    adapter_rank = 8
    model = Net(num_classes=[4, 4, 4], nb_tasks=nb_tasks, adapter_rank=adapter_rank)

    x = torch.randn(2, 3, 64, 64)
    y0 = model(x, task=0)
    y1 = model(x, task=1)
    print("task0 output:", y0.shape)
    print("task1 output:", y1.shape)