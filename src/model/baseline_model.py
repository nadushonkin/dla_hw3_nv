import numpy as np
from torch import flatten, nn, tanh
from torch.nn import Sequential
from torch.nn.utils import weight_norm


def count_padding(kernel_size, dilation):
    return (kernel_size - 1) * dilation // 2


def init_weights(module, mean=0.0, std=0.01):
    return nn.init.normal_(module.weight, mean=mean, std=std)


class ResBlock(nn.Module):
    """
    ResBlock (type 2)
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation[0],
                        padding=count_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation[1],
                        padding=count_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(lambda module: init_weights(module, mean=0.0, std=0.01))

    def forward(self, data_object, **batch):
        output = data_object
        for conv in self.convs:
            residual = output
            output = self.leaky_relu(output)
            output = conv(output)
            output += residual

        return output


class MRF(nn.Module):
    def __init__(
        self,
        mrf_id,
        num_mrfs,
        num_init_channels,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
    ):
        super().__init__()
        self.mrf_id = mrf_id
        self.num_kernels = len(resblock_kernel_sizes)
        self.resblocks = nn.ModuleList()
        for mrf_id in num_mrfs:
            num_channels_i = num_init_channels // (2 ** (mrf_id + 1))
            for _, (kernel_size_i, dilation_size_i) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    ResBlock(num_channels_i, kernel_size_i, dilation_size_i)
                )

    def forward(self, data_object, **batch):
        output = self.resblocks[self.mrf_id * self.num_kernels](data_object)
        for ind in range(self.num_kernels):
            output += self.resblocks[self.mrf_id * self.num_kernels + ind](data_object)
        return output


class Generator(nn.Module):
    def __init__(
        self,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
    ):
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = weight_norm(
            nn.Conv1d(80, upsample_initial_channel, 7, 1, padding=3)
        )
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.ConvTrs = nn.ModuleList()
        for ind, (rate_ind, kernel_size_ind) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes)
        ):
            self.ConvTrs.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**ind),
                        upsample_initial_channel // (2 ** (ind + 1)),
                        kernel_size_ind,
                        rate_ind,
                        padding=(kernel_size_ind - rate_ind) // 2,
                    )
                )
            )
        self.ConvTrs.apply(lambda module: init_weights(module, mean=0.0, std=0.01))

        self.num_mrfs = len(self.ConvTrs)
        self.MRFs = MRF(
            0,
            self.num_mrfs,
            upsample_initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
        )
        self.conv_post = weight_norm(
            nn.Conv1d(
                upsample_initial_channel // (2 ** (self.num_mrfs)), 1, 7, 1, padding=3
            )
        )
        self.conv_post.apply(init_weights)

    def forward(self, data_object, **batch):
        output = self.conv_pre(data_object)
        for ind in range(self.num_mrfs):
            output = self.leaky_relu(output)
            output = self.ConvTrs[ind](output)
            self.MRFs.mrf_id = ind
            residual = self.MRFs(output)
            output = residual / self.num_kernels
        output = self.leaky_relu(output)
        output = self.conv_post(output)
        output = tanh(output)
        return {"generated_audio": output}


class SubMPD(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        self.period = period
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.convs = nn.ModuleList([])
        channels = np.array([1, 32, 128, 512, 1024])
        channels = np.stack((channels[:-1], channels[1:]), axis=1).tolist()
        for in_out_pair in channels:
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_out_pair[0],
                        in_out_pair[1],
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(count_padding(5, 1), 0),
                    )
                )
            )
        self.convs.append(
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))
        )
        self.convs.append(weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))))

    def forward(self, data_object, **batch):
        feature_map = []

        if isinstance(self, SubMPD):
            time = data_object.shape[3]
            if time % self.period != 0:
                n_pad = self.period - (time % self.period)
                output = nn.functional.pad(data_object, (0, n_pad), "reflect")
                time += n_pad
            output = output.reshape(
                data_object.shape[0],
                data_object.shape[1],
                time // self.period,
                self.period,
            )

        for conv in self.convs[:-1]:
            output = conv(output)
            output = self.leaky_relu(output)
            feature_map.append(output)

        output = self.convs[-1](output)
        feature_map.append(output)
        output = flatten(output, 1, -1)

        return [output, feature_map]


class MPD(nn.Module):
    def __init__(self):
        self.periods = [2, 3, 5, 7, 11]

    def forward(self, output_real, output_gen, **batch):
        pred_fmap_for_real = []
        pred_fmap_for_gen = []

        for period in self.periods:
            submpd = SubMPD(period)
            pred_fmap_for_real.append(submpd(output_real))
            pred_fmap_for_gen.append(submpd(output_gen))

        return pred_fmap_for_real, pred_fmap_for_gen


class SubMSD(SubMPD):
    def __init__(self, use_spectral_norm=False):
        if use_spectral_norm:
            norm = nn.utils.spectral_norm
        else:
            norm = weight_norm

        self.convs = nn.ModuleList(
            [
                norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
                norm(nn.Conv1d(1024, 1, 3, 1, padding=1)),
            ]
        )
        self.leaky_relu = nn.LeakyReLU(0.1)


class MSD(nn.Module):
    def __init__(
        self,
    ):
        self.net_block = nn.Sequential(nn.AvgPool1d(4, 2, padding=2), SubMSD())

    def forward(self, output_real, output_gen, **batch):
        first_sub_discr = SubMSD(use_spectral_norm=True)
        pred_fmap_for_real = [first_sub_discr(output_real)]
        pred_fmap_for_gen = [first_sub_discr(output_gen)]

        for _ in range(2):
            pred_fmap_for_real.append(self.net_block(output_real))
            pred_fmap_for_gen.append(self.net_block(output_gen))

        return pred_fmap_for_real, pred_fmap_for_gen
