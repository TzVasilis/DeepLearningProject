from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AtrousConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation_rate: int,
    ) -> None:
        """
        Atrous (Dilated) Convolution Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding for the convolution.
            dilation_rate (int): Dilation rate for atrous convolution.
        """

        super(AtrousConvolutionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation_rate,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Atrous Convolution Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        return self.block(x)


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Atrous Spatial Pyramid Pooling (ASPP) Block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super(AtrousSpatialPyramidPooling, self).__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_1x1 = AtrousConvolutionBlock(
            in_channels, out_channels, 1, 0, 1
        )
        self.conv_6x6 = AtrousConvolutionBlock(
            in_channels, out_channels, 3, 6, 6
        )
        self.conv_12x12 = AtrousConvolutionBlock(
            in_channels, out_channels, 3, 12, 12
        )
        self.conv_18x18 = AtrousConvolutionBlock(
            in_channels, out_channels, 3, 18, 18
        )
        self.final_conv = AtrousConvolutionBlock(
            out_channels * 5, out_channels, 1, 0, 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ASPP Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x_1x1 = self.conv_1x1(x)
        x_6x6 = self.conv_6x6(x)
        x_12x12 = self.conv_12x12(x)
        x_18x18 = self.conv_18x18(x)

        img_pooling_output = self.pool(x)
        img_pooling_output = F.interpolate(
            img_pooling_output,
            size=x_18x18.size()[2:],
            mode="bilinear",
            align_corners=True,
        )

        concatenated_features = torch.cat(
            (x_1x1, x_6x6, x_12x12, x_18x18, img_pooling_output), dim=1
        )
        final_convolution_output = self.final_conv(concatenated_features)

        return final_convolution_output


class ResNet18(nn.Module):
    def __init__(self, output_layer: Optional[str] = None) -> None:
        """
        Modified ResNet18 model with customizable output layer.

        Args:
            output_layer (str, optional): Name of the output layer to
            extract. Default is None.
        """

        super(ResNet18, self).__init__()
        pretrained_model = models.resnet18(pretrained=True)
        self.output_layer_name = output_layer

        self.layers_up_to_output = list(pretrained_model.children())[:-1]
        if self.output_layer_name:
            self.layers_up_to_output = self.layers_up_to_output[
                : self.layers_up_to_output.index(
                    getattr(pretrained_model, self.output_layer_name)
                )
                + 1
            ]

        self.network = nn.Sequential(*self.layers_up_to_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet18 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        return self.network(x)


class Deeplabv3Plus(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        """
        Deeplabv3+ model with a ResNet18 backbone and ASPP module.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
        """

        super(Deeplabv3Plus, self).__init__()

        self.resnet18_backbone = ResNet18(output_layer="layer3")
        self.low_level_features = ResNet18(output_layer="layer1")
        self.aspp_module = AtrousSpatialPyramidPooling(
            in_channels=256, out_channels=256
        )

        self.conv1x1_layer = AtrousConvolutionBlock(64, 48, 1, 0, 1)

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.final_classifier = nn.Conv2d(256, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Deeplabv3+ model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        resnet18_output = self.resnet18_backbone(x)
        low_level_features_output = self.low_level_features(x)
        aspp_output = self.aspp_module(resnet18_output)

        aspp_upsampled = F.interpolate(
            aspp_output,
            scale_factor=(4, 4),
            mode="bilinear",
            align_corners=True,
        )
        conv1x1_output = self.conv1x1_layer(low_level_features_output)

        concatenated_input = torch.cat([conv1x1_output, aspp_upsampled], dim=1)
        conv_3x3_output = self.conv_3x3(concatenated_input)

        upscaled_3x3_output = F.interpolate(
            conv_3x3_output,
            scale_factor=(4, 4),
            mode="bilinear",
            align_corners=True,
        )
        final_output = self.final_classifier(upscaled_3x3_output)

        return final_output
