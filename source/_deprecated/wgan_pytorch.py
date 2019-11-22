import argparse
import os
import numpy as np
import math
import datetime
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training. Default: 200")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches. Default: 16")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate for optimizer. Default: 0.00005")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation. Default: 8")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space. Default: 100")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension. Default: 128")
parser.add_argument("--channels", type=int, default=3, help="number of image channels. Default: 3")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter. Default: 5")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights. Default: 0.01")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image samples. Default: 100")
parser.add_argument("--save_dir", type=str, default="WGAN", help="Where to save the models and generated images. Default:'WGAN'.")
opt = parser.parse_args()
print(opt)

save_dir = "../model/" + opt.save_dir
image_dir = save_dir + "/images"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

cuda = True if torch.cuda.is_available() else False

class artData(Dataset):
    """ Custom pytorch dataset"""

    def __init__(self, filepath):
        self.rootdir = filepath ## pytorch is channels first: [batch_size, color_channels. height, width]
        self.data = np.load(file=filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return "Dataset containts {} images".format(len(self.data))


def decrease_lr(optimizer, decrease_by=0.01):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= (1-decrease_by)

def weights_init_normal(m):
    """
    Initializes weights for weight matrices. Enhances normal distribution with zero mean and 0.02 standard deviation
    :param m: Torch model
    :return:
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size(0), real_data.nelement()/real_data.size(0)).contiguous().view(real_data.size(0), 3, 32, 32)
    alpha = alpha.cuda() if cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if cuda:
        interpolates = interpolates.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(in_features=opt.latent_dim, out_features=128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            #nn.BatchNorm2d(num_features=128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(num_features=128, eps=0.8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(num_features=64, eps=0.8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=opt.channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=False):
            block = [nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                               kernel_size=3, stride=2, padding=1),
                     nn.LeakyReLU(negative_slope=0.2, inplace=True),
                     nn.Dropout2d(p=0.25)]
            if bn:
                block.append(nn.BatchNorm2d(num_features=out_filters, eps=0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(in_features=128 * ds_size ** 2, out_features=1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        out = self.adv_layer(out)

        return out

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

print(generator)
print(discriminator)

if cuda:
    generator.cuda()
    discriminator.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
artDataset = artData(filepath="../data/train_data.npy")

artDataLoader = DataLoader(dataset=artDataset,
                           batch_size=opt.batch_size,
                           shuffle=True,
                           num_workers=0)
# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
one = torch.tensor(1.0)
mone = one*-1.0

if cuda:
    one = one.cuda()
    mone= mone.cuda()

d_loss_statistics = [None]*opt.n_epochs
g_loss_statistics = [None]*opt.n_epochs

# ----------
#  Training
# ----------
gen_iterations = 0
training_steps = 0
for epoch in range(opt.n_epochs):
    ### Training over batches:
    epoch_start_time = datetime.datetime.now().replace(microsecond=0)
    print("Epoch: {} training....".format(epoch))
    for step, batch in tqdm(enumerate(artDataLoader), total=len(artDataLoader)):
        batch = Variable(batch.type(Tensor))
        ############################
        # (1) Update D network for d_iters iterations
        ###########################
        for p in discriminator.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in generator update

        # train the discriminator d_iters times.
        # If beginning train discriminator longer or after 100 generator steps in order the disciriminator is always a little bit better than generator...
        if gen_iterations < 10 or gen_iterations % 100 == 0:
                d_iters = 100
        else:
                d_iters = opt.n_critic

        for p in discriminator.parameters():
            p.data.clamp_(min=-opt.clip_value, max=opt.clip_value)

        ## train discriminator on real and fake data
        discriminator.zero_grad()
        ## 1A train with real data
        output = discriminator(batch)
        D_x = output.mean()
        # Compute gradient
        D_x.backward(one)
        ## 1B train with fake data
        # Sampled data
        z = Variable(Tensor(np.random.normal(loc=0, scale=1, size=(batch.size(0), opt.latent_dim)))).float()
        z.requires_grad = False
        gen_imgs = generator(z).data  ## or detach()
        output = discriminator(gen_imgs)
        D_gz = output.mean()
        D_gz.backward(mone)

        ## Total discriminator loss (Wasserstein loss)
        d_loss = D_x - D_gz

        ## Perform gradient update
        optimizer_D.step()

        ## Update generator if d_iters steps have been executed:
        if training_steps % d_iters == 0:
            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = False# to avoid computation

            ## Train the generator
            optimizer_G.zero_grad()
            # Generate a batch of fake data
            gen_imgs = generator(z)
            output = discriminator(gen_imgs)
            g_loss = output.mean()
            g_loss.backward(one)
            # DO gradient update
            optimizer_G.step()

            gen_iterations += 1
            tqdm.write("*" * 100)
            tqdm.write("Epoch [%d/%d] Batch [%d/%d] Loss_D: %.5f Loss_G: %.5f D(x): %.5f D(G(z)): %.5f" % (
                epoch, opt.n_epochs, step, len(artDataLoader),
                d_loss.item(), g_loss.item(), D_x, D_gz)
                       )
            tqdm.write("*" * 100 + "\n")

        if training_steps % opt.sample_interval == 0:
            save_image(gen_imgs.data[:16], image_dir + "/%d.png" % training_steps, nrow=4, normalize=True)

        training_steps += 1

    epoch_time_difference = datetime.datetime.now().replace(microsecond=0) - epoch_start_time
    tqdm.write("Epoch: {:3d} time execution: {}".format(epoch, epoch_time_difference))


    ## decrease learning rate every epoch
    decrease_lr(optimizer_G, decrease_by=0.001)
    decrease_lr(optimizer_D, decrease_by=0.001)

    ## Save generator and discriminator every (
    if epoch % 500 ==0 or epoch == opt.n_epochs-1:
        torch.save(generator.state_dict(), save_dir + "/generator_epoch{}.ckpt".format(
            epoch))
        torch.save(discriminator.state_dict(), save_dir + "/discriminator_epoch{}.ckpt".format(
            epoch))