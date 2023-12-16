import torch
import torch.nn as nn


def initialize_weights(module: nn.Module) -> None:
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


class DoubleConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int
    ) -> None:
        """
        A block containing two convolutional layers with batch
        normalization and ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
        """

        super(DoubleConvBlock, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.process.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the double convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        return self.process(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 9) -> None:
        """
        U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super(UNet, self).__init__()
        # Encoder.
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DoubleConvBlock(
            in_channels=in_channels, out_channels=64, kernel_size=3
        )
        self.enc2 = DoubleConvBlock(
            in_channels=64, out_channels=128, kernel_size=3
        )
        self.enc3 = DoubleConvBlock(
            in_channels=128, out_channels=256, kernel_size=3
        )
        self.enc4 = DoubleConvBlock(
            in_channels=256, out_channels=512, kernel_size=3
        )
        self.enc5 = DoubleConvBlock(
            in_channels=512, out_channels=1024, kernel_size=3
        )

        # Decoder.
        self.up_conv1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.up_conv2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.up_conv3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.up_conv4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )

        self.dec1 = DoubleConvBlock(
            in_channels=1024, out_channels=512, kernel_size=3
        )
        self.dec2 = DoubleConvBlock(
            in_channels=512, out_channels=256, kernel_size=3
        )
        self.dec3 = DoubleConvBlock(
            in_channels=256, out_channels=128, kernel_size=3
        )
        self.dec4 = DoubleConvBlock(
            in_channels=128, out_channels=64, kernel_size=3
        )

        self.output = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        skip_connections = []

        # Encoder.
        x1 = self.enc1(x)
        skip_connections.append(x1)
        x2 = self.enc2(self.maxpool(x1))
        skip_connections.append(x2)
        x3 = self.enc3(self.maxpool(x2))
        skip_connections.append(x3)
        x4 = self.enc4(self.maxpool(x3))
        skip_connections.append(x4)
        x5 = self.enc5(self.maxpool(x4))

        # Decoder.
        x = self.up_conv1(x5)
        y = skip_connections[3]
        x = torch.nn.functional.interpolate(x, size=y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.dec1(y_new)

        x = self.up_conv2(x)
        y = skip_connections[2]
        x = torch.nn.functional.interpolate(x, size=y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.dec2(y_new)

        x = self.up_conv3(x)
        y = skip_connections[1]
        x = torch.nn.functional.interpolate(x, size=y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.dec3(y_new)

        x = self.up_conv4(x)
        y = skip_connections[0]
        x = torch.nn.functional.interpolate(x, size=y.shape[2:])
        y_new = torch.cat((y, x), dim=1)
        x = self.dec4(y_new)

        x = self.output(x)
        return x
