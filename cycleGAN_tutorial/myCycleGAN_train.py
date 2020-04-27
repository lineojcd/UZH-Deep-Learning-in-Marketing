import sys
# Append the path to the folder holding the scripts
# if it is not already the working directory
sys.path.append('.')

import os
import itertools
from math import floor

import imageio

import numpy as np
import time

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator, Discriminator
from utils import ReplayBuffer, LambdaLR, weights_init_normal, safe_mkdirs, tensor2image, as_np
from datasets import ImageDataset

load_iter = 0      # Starting iteration (if 0 train from scratch)
n_epochs = 200     # Number of epochs of training
batch_size = 1     # Size of the batches
lr = 0.0002        # Initial learning rate
decay_epoch = 100  # Epoch to start linearly decaying the learning rate to 0
size = 128         # Size of the data crop (squared assumed)

input_nc = 3       # Number of channels of input data
output_nc = 3      # Number of channels of output data
n_cpu = 2          # Number of cpu threads to use during batch generation
image_size = 64    # Size of images

log_interval = 200         # Interval to print output
model_save_interval = 1000  # Interval to save model weights
image_save_interval = 1000  # Interval at which to log visual progess

# Paths to directories
ROOT = '.'  # Change if necessary
data_path = os.path.join(ROOT, 'data')
output_path = os.path.join(ROOT, 'output_my')
output_imgs_path = os.path.join(output_path, 'imgs')
output_weights_path = os.path.join(output_path, 'weights')
safe_mkdirs(output_path)
safe_mkdirs(output_imgs_path)
safe_mkdirs(output_weights_path)

# Use Cuda if available. It should be available on colab.
use_cuda = torch.cuda.is_available()

# From paper: We use 6 residual blocks for 128 × 128 training images,
# and 9 residual blocks for 256 × 256 or higher-resolution training images

# in our code: defult setting in is 3
n_res_block = 6

# Initialize networks
# Domain A: dresses
# Domain B: shoes
netG_A2B = Generator(input_nc, output_nc,n_residual_blocks=n_res_block)
netG_B2A = Generator(output_nc, input_nc,n_residual_blocks=n_res_block)
# netG_A2B = Generator(input_nc, output_nc)
# netG_B2A = Generator(output_nc, input_nc)
netD_A = Discriminator(input_nc)
netD_B = Discriminator(output_nc)

# To cuda
if use_cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

# Initialize or load pretrained weights
# TODO: make sure this works. I have found the save/load_state_dict API to be buggy
# so make sure it works on your machine before you spend a lot of time
# training a model. If you save a corrupted weights, you can't restore them.
if load_iter == 0:
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
else:
    netG_A2B.load_state_dict(torch.load(os.path.join(output_weights_path, 'netG_A2B_i{}.0.pth'.format(load_iter*100))))
    netG_B2A.load_state_dict(torch.load(os.path.join(output_weights_path, 'netG_B2A_i{}.0.pth'.format(load_iter*100))))
    netD_A.load_state_dict(torch.load(os.path.join(output_weights_path, 'netD_A_i{}.0.pth'.format(load_iter*100))))
    netD_B.load_state_dict(torch.load(os.path.join(output_weights_path, 'netD_B_i{}.0.pth'.format(load_iter*100))))

    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()

# Losses
# TODO: Defining the correct losses make and break the performance of GANs.
#       Can you think of a loss that would improve the results?

# Notice that there are three losses here
# 1) criterion_GAN: This is the standard GAN loss. Whether the discriminator
#    could correctly predict whether the image was real or fake is used to train the networks.
#    In paper: the negative log likelihood objective is replaced by a least-squares loss.
#    This loss is more stable during training and generates higher quality results.
# 2) criterion_cycle: cycle-consistency discussed in the intro text.
# 3) criterion_identity: if you put an image of a shoe into the dress to
#    shoe generator, you should get the same image of the shoe back.
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers (with decay)
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

start_epoch = floor(load_iter / 995540)
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if use_cuda else torch.Tensor
input_A = Tensor(batch_size, input_nc, image_size, image_size)
input_B = Tensor(batch_size, output_nc, image_size, image_size)
target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

# Initialize replay buffer
# The replay buffer helps stabilize training by saving images of previous
# iterations and using them as training data in later iterations. This helps
# avoid over training on current data.

# In paper: the default max replay buffer size is 50
max_replay = 100

fake_A_buffer = ReplayBuffer(max_size=max_replay)
fake_B_buffer = ReplayBuffer(max_size=max_replay)

# fake_A_buffer = ReplayBuffer()
# fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.Resize(image_size, Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# For training set
# The `ImageDataset` is in dataset.py. Check it out to see what it does.
dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_, unaligned=True),
                        batch_size=batch_size, shuffle=True, num_workers=n_cpu)

# For test set
dataloader_test = DataLoader(ImageDataset(data_path, transforms_=transforms_, mode='test'),
                             batch_size=1, shuffle=True, num_workers=n_cpu)

prev_time = time.time()
iter = load_iter

# Training
for epoch in range(start_epoch, n_epochs):
    print('current epoch:', epoch)
    dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_, unaligned=True),
                            batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    for i, batch in enumerate(dataloader):
        if i % 100 == 0:
            if load_iter ==0 :
                print( "training from scratch:",i)
            else:
                print("resume training:", i + load_iter*100)
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Forward pass through each of the generators and discriminators
        # Generators A2B and B2A ##############################################
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

        # Total loss
        loss_G = (loss_identity_A + loss_identity_B) * 5.0
        loss_G += (loss_GAN_A2B + loss_GAN_B2A) * 1.0
        loss_G += (loss_cycle_ABA + loss_cycle_BAB) * 10.0

        # Calculate the gradient of the generators
        loss_G.backward()

        # Update the weights of the generators
        optimizer_G.step()

        # Discriminator A #####################################################
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5

        # Calculate the gradient of discriminator A
        loss_D_A.backward()

        # Update the weights of discriminator A
        optimizer_D_A.step()

        # Discriminator B #####################################################
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5

        # Calculate the gradient of discriminator B
        loss_D_B.backward()

        # Update the weights of discriminator B
        optimizer_D_B.step()

        # Track performance
        if iter % log_interval == 0:
            print('---------------------')
            print('GAN loss:', as_np(loss_GAN_A2B), as_np(loss_GAN_B2A))
            print('Identity loss:', as_np(loss_identity_A), as_np(loss_identity_B))
            print('Cycle loss:', as_np(loss_cycle_ABA), as_np(loss_cycle_BAB))
            print('D loss:', as_np(loss_D_A), as_np(loss_D_B))
            print('time:', time.time() - prev_time)
            prev_time = time.time()

        # Print outputs
        if iter % image_save_interval == 0:
            output_path_ = os.path.join(output_imgs_path, str(iter  + load_iter*100)+'f')
            safe_mkdirs(output_path_)

            for j, batch_ in enumerate(dataloader_test):

                if j < 60:
                    real_A_test = Variable(input_A.copy_(batch_['A']))
                    real_B_test = Variable(input_B.copy_(batch_['B']))

                    fake_AB_test = netG_A2B(real_A_test)
                    fake_BA_test = netG_B2A(real_B_test)

                    recovered_ABA_test = netG_B2A(fake_AB_test)
                    recovered_BAB_test = netG_A2B(fake_BA_test)

                    fn = os.path.join(output_path_, str(j))
                    A_test = np.hstack([tensor2image(real_A_test[0]), tensor2image(fake_AB_test[0]),
                                        tensor2image(recovered_ABA_test[0])])
                    B_test = np.hstack([tensor2image(real_B_test[0]), tensor2image(fake_BA_test[0]),
                                        tensor2image(recovered_BAB_test[0])])
                    imageio.imwrite(fn + '_A.jpg', A_test)
                    imageio.imwrite(fn + '_B.jpg', B_test)

                    # imageio.imwrite(fn + '.A.jpg', tensor2image(real_A_test[0]))
                    # imageio.imwrite(fn + '.B.jpg', tensor2image(real_B_test[0]))
                    # imageio.imwrite(fn + '.BA.jpg', tensor2image(fake_BA_test[0]))
                    # imageio.imwrite(fn + '.AB.jpg', tensor2image(fake_AB_test[0]))
                    # imageio.imwrite(fn + '.ABA.jpg', tensor2image(recovered_ABA_test[0]))
                    # imageio.imwrite(fn + '.BAB.jpg', tensor2image(recovered_BAB_test[0]))
                else:
                    break

        # Save models checkpoints
        if iter % model_save_interval == 0:
            torch.save(netG_A2B.state_dict(), os.path.join(output_weights_path, 'fnetG_A2B_i{}.pth'.format(iter + load_iter*100)))
            torch.save(netG_B2A.state_dict(), os.path.join(output_weights_path, 'fnetG_B2A_i{}.pth'.format(iter+ load_iter*100)))
            torch.save(netD_A.state_dict(), os.path.join(output_weights_path, 'fnetD_A_i{}.pth'.format(iter + load_iter*100)))
            torch.save(netD_B.state_dict(), os.path.join(output_weights_path, 'fnetD_B_i{}.pth'.format(iter + load_iter*100)))

        iter += 1

        if i >19999:
            break

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()


