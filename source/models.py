import torch.nn as nn
import torch
from spectral_norm import SpectralNorm
# Define custom Transposed 2D convolution with upsampling layer or without
class Transpose2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upsample):
        super(Transpose2DLayer, self).__init__()
        self.upsample = upsample
        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode='nearest')
        self.conv2dTrans = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
            return self.conv2dTrans((self.upsample_layer(x)))

# Generator model modified from DCGAN paper functional without nn.Sequential
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(in_features=self.latent_dim, out_features=4*4*1024)
        self.transConv1 = Transpose2DLayer(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1,
                                           upsample=2)
        self.batchNorm1 = nn.BatchNorm2d(num_features=512)
        self.transConv2 = Transpose2DLayer(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1,
                                           upsample=2)
        self.batchNorm2 = nn.BatchNorm2d(num_features=256)
        self.transConv3 = Transpose2DLayer(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1,
                                           upsample=2)
        self.batchNorm3 = nn.BatchNorm2d(num_features=128)
        self.transConv4 = Transpose2DLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1,
                                           upsample=2)
        self.batchNorm4 = nn.BatchNorm2d(num_features=64)
        self.transConv5 = Transpose2DLayer(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1,
                                           upsample=2)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 4, 4)
        x = self.transConv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.transConv2(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.transConv3(x)
        x = self.batchNorm3(x)
        x = self.relu(x)
        x = self.transConv4(x)
        x = self.batchNorm4(x)
        x = self.relu(x)
        x = self.transConv5(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, nchannels=3):
        super(Discriminator, self).__init__()

        # in: (3,128,128)
        self.conv1 = SpectralNorm(module=nn.Conv2d(in_channels=nchannels, out_channels=64,
                                                   kernel_size=3, stride=1, padding=1))
        self.conv2 = SpectralNorm(module=nn.Conv2d(in_channels=64, out_channels=64,
                                                   kernel_size=3, stride=2, padding=1))
        # out: (64,64,64)

        # in: (64,64,64)
        self.conv3 = SpectralNorm(module=nn.Conv2d(in_channels=64, out_channels=128,
                                                   kernel_size=3, stride=1, padding=1))
        self.conv4 = SpectralNorm(module=nn.Conv2d(in_channels=128, out_channels=128,
                                                   kernel_size=4, stride=2, padding=1))
        # out: (128,32,32)

        # in: (128,32,32)
        self.conv5 = SpectralNorm(module=nn.Conv2d(in_channels=128, out_channels=256,
                                                   kernel_size=3, stride=1, padding=1))
        self.conv6 = SpectralNorm(module=nn.Conv2d(in_channels=256, out_channels=256,
                                                   kernel_size=4, stride=2, padding=1))

        # out: (256,16,16)
        self.conv7 = SpectralNorm(module=nn.Conv2d(in_channels=256, out_channels=512,
                                                   kernel_size=3, stride=1, padding=1))
        # out: (512, 16, 16)

        # in: (512*16*16, )
        self.linear = nn.Linear(in_features=512*16*16, out_features=1)
        # out (1, )

        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(inplace=True, negative_slope=0.1)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = x.view(-1, 512*16*16)
        x = self.linear(x)
        x = self.sigmoid(x)

        return x

"""
from torchvision.utils import save_image
batch = torch.randn((25, 100))
generator = Generator()
generated_batch = generator(batch)
print(generated_batch.size())
discriminator = Discriminator()
output_batch = discriminator(generated_batch)
output_batch.size()
save_image(generated_batch.data[:25], 'test.png', nrow=5, normalize=False)
"""

if __name__ == '__main__':
    generator = Generator()
    print(generator)
    discriminator = Discriminator()
    print(discriminator)
