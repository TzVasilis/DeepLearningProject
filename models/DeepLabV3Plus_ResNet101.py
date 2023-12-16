from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AtrousConvolution(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int,
        dilation_rate: int,
    ) -> None:
        """
        Atrous Convolution block.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding added to the input.
            dilation_rate (int): Dilation rate for atrous convolution.
        """

        super(AtrousConvolution, self).__init__()

        self.process = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation_rate,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Atrous Convolution block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        return self.process(x)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Atrous Spatial Pyramid Pooling (ASPP) module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super(ASPP, self).__init__()

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_1x1 = AtrousConvolution(in_channels, out_channels, 1, 0, 1)
        self.conv_6x6 = AtrousConvolution(in_channels, out_channels, 3, 6, 6)
        self.conv_12x12 = AtrousConvolution(
            in_channels, out_channels, 3, 12, 12
        )
        self.conv_18x18 = AtrousConvolution(
            in_channels, out_channels, 3, 18, 18
        )
        self.final_conv = AtrousConvolution(
            out_channels * 5, out_channels, 1, 0, 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ASPP module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x_1x1 = self.conv_1x1(x)
        x_6x6 = self.conv_6x6(x)
        x_12x12 = self.conv_12x12(x)
        x_18x18 = self.conv_18x18(x)

        img_pool_opt = self.pool(x)
        img_pool_opt = F.interpolate(
            img_pool_opt,
            size=x_18x18.size()[2:],
            mode="bilinear",
            align_corners=True,
        )

        concat = torch.cat(
            (x_1x1, x_6x6, x_12x12, x_18x18, img_pool_opt), dim=1
        )
        x_final_conv = self.final_conv(concat)

        return x_final_conv


class ModifiedResNet101(nn.Module):
    def __init__(self, output_layer: Optional[str] = None) -> None:
        """
        Modified ResNet101 model.

        Args:
            output_layer (str, optional): Name of the output layer in
            ResNet101. Default is None.
        """

        super(ModifiedResNet101, self).__init__()
        self.pretrained = models.resnet101(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Modified ResNet101 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x = self.net(x)
        return x


class Deeplabv3Plus(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        """
        Deeplabv3Plus model.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
        """

        super(Deeplabv3Plus, self).__init__()

        self.backbone = ModifiedResNet101(output_layer="layer3")
        self.low_level_features = ModifiedResNet101(output_layer="layer1")
        self.assp = ASPP(in_channels=1024, out_channels=256)

        self.conv1x1 = AtrousConvolution(256, 48, 1, 0, 1)

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(256, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Deeplabv3Plus model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x_backbone = self.backbone(x)
        x_low_level = self.low_level_features(x)
        x_assp = self.assp(x_backbone)

        x_assp_upsampled = F.interpolate(
            x_assp, scale_factor=(4, 4), mode="bilinear", align_corners=True
        )
        x_conv1x1 = self.conv1x1(x_low_level)

        x_cat = torch.cat([x_conv1x1, x_assp_upsampled], dim=1)
        x_3x3 = self.conv_3x3(x_cat)

        x_3x3_upscaled = F.interpolate(
            x_3x3, scale_factor=(4, 4), mode="bilinear", align_corners=True
        )
        x_out = self.classifier(x_3x3_upscaled)

        return x_out
