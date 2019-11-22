import argparse
import os
import numpy as np
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

import torch.nn as nn
import torch
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--max_lr_decrease_epoch", type=int, default=500, help="Maximum epoch iteration until learning rate decrease should be done")
parser.add_argument("--d_lr", type=float, default=0.000004, help="adam: discriminator learning rate")
parser.add_argument("--g_lr", type=float, default=0.00004, help="adam: generator learning rate")
parser.add_argument("--ngf", type=int, default=64, help="Size of feature map in generator")
parser.add_argument("--ndf", type=int, default=128, help="Size of feature map in discrininator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--save_dir", type=str, default="dcgan", help="Where the model and generated images should be saved. Defaults to 'DCGAN'")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
parser.add_argument("--load_generator", type=str, default=None, help="Model checkpoint for generator to start training again. Default: None")
parser.add_argument("--load_discriminator", type=str, default=None, help="Model checkpoint for discriminator to start training again. Default: None")
parser.add_argument("--last_epoch", type=int, default=0, help="If model_checkpoint exists, last epoch should be inserted to continue training. Default: 0")
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


def decrease_lr(optimizer, decrease_by=0.01):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= (1-decrease_by)

class Generator(nn.Module):
    def __init__(self,):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=opt.latent_dim, out_channels=opt.ngf * 16, kernel_size=4, stride=1, padding=0,
                               bias=False),
            #nn.BatchNorm2d(num_features=opt.ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4

            nn.ConvTranspose2d(in_channels=opt.ngf * 16, out_channels=opt.ngf * 8, kernel_size=4, stride=2, padding=1,
                               bias=False),
            #nn.BatchNorm2d(num_features=opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8

            nn.ConvTranspose2d(in_channels=opt.ngf * 8, out_channels=opt.ngf * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            #nn.BatchNorm2d(num_features=opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16

            nn.ConvTranspose2d(in_channels=opt.ngf * 4, out_channels=opt.ngf * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            #nn.BatchNorm2d(num_features=opt.ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32

            nn.ConvTranspose2d(in_channels=opt.ngf * 2, out_channels=opt.ngf * 1, kernel_size=4, stride=2, padding=1,
                               bias=False),
            #nn.BatchNorm2d(num_features=opt.ngf * 1),
            nn.ReLU(True),
            # state size. (ngf*1) x 64 x 64

            nn.ConvTranspose2d(in_channels=opt.ngf, out_channels=3, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
            # state size. (3) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(in_channels=3, out_channels=opt.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64

            nn.Conv2d(in_channels=opt.ndf, out_channels=opt.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(num_features=opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32

            nn.Conv2d(in_channels=opt.ndf * 2, out_channels=opt.ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(num_features=opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16

            nn.Conv2d(in_channels=opt.ndf * 4, out_channels=opt.ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(num_features=opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8

            nn.Conv2d(in_channels=opt.ndf * 8, out_channels=opt.ndf * 16, kernel_size=4, stride=2, padding=1,
                      bias=False),
            #nn.BatchNorm2d(num_features=opt.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4

            nn.Conv2d(in_channels=opt.ndf * 16, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Loss function
adversarial_loss = torch.nn.BCELoss()
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if opt.load_generator is not None:
    generator_path = "../model/{}/{}".format(opt.save_dir, opt.load_generator)
    try:
        print("loading generator..: {}".format(generator_path))
        generator.load_state_dict(torch.load(generator_path))
    except:
        print("Couldn't load generator model {}".format(generator_path))
        quit()
if opt.load_discriminator is not None:
    discriminator_path = "../model/{}/{}".format(opt.save_dir, opt.load_discriminator)
    try:
        print("loading discriminator..: {}".format(discriminator_path))
        discriminator.load_state_dict(torch.load(discriminator_path))
    except:
        print("Couldn't load discriminator model {}".format(discriminator_path))
        quit()
print(generator)
print(discriminator)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

if opt.load_generator is None and opt.load_discriminator is None:
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
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if opt.n_epochs < 100:
    save_epoch = opt.n_epochs // 10
elif opt.n_epochs < 200:
    save_epoch = opt.n_epochs // 50
elif opt.n_epochs < 500:
    save_epoch = opt.n_epochs // 100
else:
    save_epoch = opt.n_epochs // 250

d_loss_stats = []
dx_stats = []
dgz1_stats = []
dgz2_stats = []
g_loss_stats = []
g_lr_stats = []
d_lr_stats = []
# ----------
#  Training
# ----------
print("Start training...")
for epoch in range(opt.last_epoch, opt.last_epoch+opt.n_epochs, 1):
    d_loss_b = []
    dx_b = []
    dgz1_b = []
    dgz2_b = []
    g_loss_b = []
    for step, batch in tqdm(enumerate(artDataLoader), total=len(artDataLoader)):
        batch = Variable(batch.type(Tensor))
        # Adversarial ground truths
        valid = Variable(Tensor(batch.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch.size(0), 1).fill_(0.0), requires_grad=False)
        # Sampled data
        z = Variable(Tensor(np.random.normal(loc=0, scale=1, size=(batch.size(0), opt.latent_dim, 1, 1)))).float()
        z.requires_grad = False

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train Discriminator on real and fake:
        discriminator.zero_grad()
        # 1A Train D on real
        output = discriminator(batch).view(-1)
        d_real_loss = adversarial_loss(output, valid)
        d_real_loss.backward()
        D_x = output.data.mean()

        # 1B: Train D on fake
        fake_data = generator(z)
        output = discriminator(fake_data.detach()).view(-1)
        d_fake_loss = adversarial_loss(output, fake)
        d_fake_loss.backward()
        D_G_z1 = output.data.mean()

        d_loss = d_real_loss + d_fake_loss
        optimizer_D.step()

        ## Train Generator: Train G on D's response (but DO NOT train D on these labels)
        ############################
        # (2) Update G network: maximize log(D(G(z))) bc. of vanishing gradient from generator
        ###########################
        generator.zero_grad()
        output = discriminator(fake_data).view(-1)
        g_loss = adversarial_loss(output, valid)  # Train generator to pretend its genuine
        g_loss.backward()
        ## should be smaller than D_G_z1 since discriminator has learned to discover fake. in d_fake_loss
        D_G_z2 = output.data.mean()
        optimizer_G.step()

        ## save stats within batch loop
        dx_b.append(D_x.item())
        dgz1_b.append(D_G_z1.item())
        dgz2_b.append(D_G_z2.item())
        d_loss_b.append(d_loss.item())
        g_loss_b.append(g_loss.item())

        ## Every 10 batches print out steps print some information
        if step % 10 == 0:
            tqdm.write("*" * 100)
            tqdm.write("Epoch [%d/%d] Batch [%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f" % (
                epoch, opt.last_epoch  + opt.n_epochs, step, len(artDataLoader),
                d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
            tqdm.write("*" * 100 + "\n")

        batches_done = epoch * len(artDataLoader) + step
        if batches_done % opt.sample_interval == 0:
            save_image(fake_data.data[:16], image_dir + "/%d.png" % batches_done, nrow=4, normalize=True)

    ## save stats
    d_lr_stats.append(opt.d_lr * (1 - 0.0001) ** epoch)
    g_lr_stats.append(opt.g_lr * (1 - 0.0001) ** epoch)
    d_loss_stats.append(np.mean(d_loss_b))
    g_loss_stats.append(np.mean(g_loss_b))
    dx_stats.append(np.mean(dx_b))
    dgz1_stats.append(np.mean(dgz1_b))
    dgz2_stats.append(np.mean(dgz2_b))

    ## decrease learning rate every epoch
    if epoch < opt.max_lr_decrease_epoch:
        decrease_lr(optimizer_G, decrease_by=0.000002)
        decrease_lr(optimizer_D, decrease_by=0.000001)

    ## Save generator and discriminator
    if epoch % save_epoch == 0 or epoch == opt.n_epochs-1:
        torch.save(generator.state_dict(), save_dir + "/generator_epoch{}.ckpt".format(
            epoch))
        torch.save(discriminator.state_dict(), save_dir + "/discriminator_epoch{}.ckpt".format(
            epoch))

np.save(file=save_dir+"/g_lr.npy", arr=g_lr_stats)
np.save(file=save_dir+"/d_lr.npy", arr=d_lr_stats)
np.save(file=save_dir+"/d_x.npy", arr=dx_stats)
np.save(file=save_dir+"/d_gz1.npy", arr=dgz1_stats)
np.save(file=save_dir+"/d_gz2.npy", arr=dgz2_stats)
np.save(file=save_dir+"/d_loss.npy", arr=d_loss_stats)
np.save(file=save_dir+"/g_loss.npy", arr=g_loss_stats)

