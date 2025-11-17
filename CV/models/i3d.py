import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super().forward(x)


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
    ):
        super().__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,
            bias=self._use_bias,
        )

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        batch, channel, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1))
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture."""

    def __init__(self, num_classes=400, spatial_squeeze=True, in_channels=3, dropout_keep_prob=0.5):
        super().__init__()

        self.num_classes = num_classes
        self.spatial_squeeze = spatial_squeeze

        self.Conv3d_1a_7x7 = Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            name="Conv3d_1a_7x7",
        )
        self.MaxPool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2))
        self.Conv3d_2b_1x1 = Unit3D(
            in_channels=64,
            output_channels=64,
            kernel_shape=[1, 1, 1],
            name="Conv3d_2b_1x1",
        )
        self.Conv3d_2c_3x3 = Unit3D(
            in_channels=64,
            output_channels=192,
            kernel_shape=[3, 3, 3],
            name="Conv3d_2c_3x3",
        )
        self.MaxPool3d_3a_3x3 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2))

        self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32], "Mixed_3b")
        self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64], "Mixed_3c")
        self.MaxPool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2))

        self.Mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64], "Mixed_4b")
        self.Mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64], "Mixed_4c")
        self.Mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64], "Mixed_4d")
        self.Mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 64], "Mixed_4e")
        self.Mixed_4f = InceptionModule(528, [256, 160, 320, 32, 128, 128], "Mixed_4f")
        self.MaxPool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2))

        self.Mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128], "Mixed_5b")
        self.Mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128], "Mixed_5c")

        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)

        self.logits = Unit3D(
            in_channels=1024,
            output_channels=self.num_classes,
            kernel_shape=[1, 1, 1],
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

    def replace_logits(self, num_classes):
        self.num_classes = num_classes
        self.logits = Unit3D(
            in_channels=1024,
            output_channels=self.num_classes,
            kernel_shape=[1, 1, 1],
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

    def forward(self, x):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)

        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)

        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.logits(x)

        if self.spatial_squeeze:
            x = x.squeeze(3).squeeze(3)

        x = x.mean(2)
        return x
