import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features, act='relu'):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]
        conv_block += [nn.ReLU(inplace=True)] if act == 'relu' else [Mish()]
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(in_features, in_features, 3),
                       nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=3,
                 img_size=64, act='relu'):
        super(Generator, self).__init__()

        filter_dim, n_padding = 3, 1

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features*2

        for _ in range(2):
            
            model += [nn.Conv2d(in_features, out_features, filter_dim, stride=2, padding=n_padding),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]

            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features, act)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]

            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc, x_size=None):
        """Initialize discriminator.

        shape:
            KEY: b: batch size; c: n channels; n: w, h of square image.
            input:      b x   c x       n x       n
            conv1:      b x  64 x     n/2 x     n/2
            conv2:      b x 128 x     n/4 x     n/4
            conv2e:     b x 128 x     n/4 x     n/4
            conv3:      b x 256 x     n/8 x     n/8
            conv3e:     b x 256 x     n/8 x     n/8
            conv4:      b x 512 x n/8 - 1 x n/8 - 1
            conv4e:     b x 512 x n/8 - 1 x n/8 - 1
            conv5:      b x   1 x n/8 - 2 x n/8 - 2
            avg_pool2d: b x   1
        """
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(input_nc, 64, 4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, 4, padding=1)
        self.in4 = nn.InstanceNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(512, 1, 4, padding=1)

    def forward(self, x):
        # x = self.model(x)
        # # Average pooling and flatten
        # return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)  # return flattened avg_pool2d

        conv1 = self.conv1(x)
        relu1 = self.relu1(conv1)

        conv2 = self.conv2(relu1)
        in2 = self.in2(conv2)
        relu2 = self.relu2(in2)

        conv3 = self.conv3(relu2)
        in3 = self.in3(conv3)
        relu3 = self.relu3(in3)

        conv4 = self.conv4(relu3)
        in4 = self.in4(conv4)
        relu4 = self.relu4(in4)

        conv5 = self.conv5(relu4)
        out = F.avg_pool2d(conv5, conv5.size()[2:]).view(conv5.size(0), -1)

        return out
