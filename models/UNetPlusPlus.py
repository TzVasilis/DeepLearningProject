import torch
import torch.nn as nn


def initialize_weights(module: torch.nn.Module) -> None:
    """
    Initialize the weights according to He et al., 2015.

    Args:
        module (torch.nn.Module): The module to initialize the weights
        for.
    """

    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(
            module.weight, mode="fan_out", nonlinearity="relu"
        )
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class ConvolutionBlock(nn.Module):
    def __init__(
        self, in_channels: int, mid_channels: int, out_channels: int
    ) -> None:
        """
        A block containing two convolutional layers with batch
        normalization and ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of channels in the intermediate
            convolutional layer.
            out_channels (int): Number of output channels.
        """

        super(ConvolutionBlock, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=True
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=1, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.process.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolution block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size,
            in_channels, height, width).

        Returns:
            torch.Tensor: Processed output tensor.
        """

        return self.process(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 9) -> None:
        """
        U-Net++ architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super(UNetPlusPlus, self).__init__()
        num_filters = [64, 128, 256, 512, 1024]

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.conv_block0_0 = ConvolutionBlock(
            in_channels, num_filters[0], num_filters[0]
        )
        self.conv_block1_0 = ConvolutionBlock(
            num_filters[0], num_filters[1], num_filters[1]
        )
        self.conv_block2_0 = ConvolutionBlock(
            num_filters[1], num_filters[2], num_filters[2]
        )
        self.conv_block3_0 = ConvolutionBlock(
            num_filters[2], num_filters[3], num_filters[3]
        )
        self.conv_block4_0 = ConvolutionBlock(
            num_filters[3], num_filters[4], num_filters[4]
        )

        self.conv_block0_1 = ConvolutionBlock(
            num_filters[0] + num_filters[1], num_filters[0], num_filters[0]
        )
        self.conv_block1_1 = ConvolutionBlock(
            num_filters[1] + num_filters[2], num_filters[1], num_filters[1]
        )
        self.conv_block2_1 = ConvolutionBlock(
            num_filters[2] + num_filters[3], num_filters[2], num_filters[2]
        )
        self.conv_block3_1 = ConvolutionBlock(
            num_filters[3] + num_filters[4], num_filters[3], num_filters[3]
        )

        self.conv_block0_2 = ConvolutionBlock(
            num_filters[0] * 2 + num_filters[1], num_filters[0], num_filters[0]
        )
        self.conv_block1_2 = ConvolutionBlock(
            num_filters[1] * 2 + num_filters[2], num_filters[1], num_filters[1]
        )
        self.conv_block2_2 = ConvolutionBlock(
            num_filters[2] * 2 + num_filters[3], num_filters[2], num_filters[2]
        )

        self.conv_block0_3 = ConvolutionBlock(
            num_filters[0] * 3 + num_filters[1], num_filters[0], num_filters[0]
        )
        self.conv_block1_3 = ConvolutionBlock(
            num_filters[1] * 3 + num_filters[2], num_filters[1], num_filters[1]
        )

        self.conv_block0_4 = ConvolutionBlock(
            num_filters[0] * 4 + num_filters[1], num_filters[0], num_filters[0]
        )

        self.final_conv = nn.Conv2d(
            num_filters[0], out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet++ model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size,
            in_channels, height, width).

        Returns:
            torch.Tensor: Processed output tensor.
        """

        x0_0 = self.conv_block0_0(x)
        x1_0 = self.conv_block1_0(self.pooling(x0_0))
        x0_1 = self.conv_block0_1(torch.cat([x0_0, self.upsampling(x1_0)], 1))

        x2_0 = self.conv_block2_0(self.pooling(x1_0))
        x1_1 = self.conv_block1_1(torch.cat([x1_0, self.upsampling(x2_0)], 1))
        x0_2 = self.conv_block0_2(
            torch.cat([x0_0, x0_1, self.upsampling(x1_1)], 1)
        )

        x3_0 = self.conv_block3_0(self.pooling(x2_0))
        x2_1 = self.conv_block2_1(torch.cat([x2_0, self.upsampling(x3_0)], 1))
        x1_2 = self.conv_block1_2(
            torch.cat([x1_0, x1_1, self.upsampling(x2_1)], 1)
        )
        x0_3 = self.conv_block0_3(
            torch.cat([x0_0, x0_1, x0_2, self.upsampling(x1_2)], 1)
        )

        x4_0 = self.conv_block4_0(self.pooling(x3_0))
        x3_1 = self.conv_block3_1(torch.cat([x3_0, self.upsampling(x4_0)], 1))
        x2_2 = self.conv_block2_2(
            torch.cat([x2_0, x2_1, self.upsampling(x3_1)], 1)
        )
        x1_3 = self.conv_block1_3(
            torch.cat([x1_0, x1_1, x1_2, self.upsampling(x2_2)], 1)
        )
        x0_4 = self.conv_block0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsampling(x1_3)], 1)
        )

        output = self.final_conv(x0_4)
        return output
