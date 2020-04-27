import argparse
import sys
import os
import numpy as np

import imageio
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from PIL import Image

from models import Generator
from datasets import ImageDataset
from utils import ReplayBuffer, LambdaLR, weights_init_normal, safe_mkdirs, tensor2image, as_np

load_iter = 70000      # Starting iteration (if 0 train from scratch)
batch_size = 1     # Size of the batches
size = 128         # Size of the data crop (squared assumed)

input_nc = 3       # Number of channels of input data
output_nc = 3      # Number of channels of output data
n_cpu = 2          # Number of cpu threads to use during batch generation
image_size = 64    # Size of images


# Paths to directories
ROOT = '.'  # Change if necessary
data_path = os.path.join(ROOT, 'data')
output_path = os.path.join(ROOT, 'output_my')
output_imgs_path = os.path.join(output_path, 'test_imgs')
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

# To cuda
if use_cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

netG_A2B.load_state_dict(torch.load(os.path.join(output_weights_path, 'fnetG_A2B_i{}.pth'.format(load_iter))))
netG_B2A.load_state_dict(torch.load(os.path.join(output_weights_path, 'fnetG_B2A_i{}.pth'.format(load_iter))))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if use_cuda else torch.Tensor
input_A = Tensor(batch_size, input_nc, image_size, image_size)
input_B = Tensor(batch_size, output_nc, image_size, image_size)

# Dataset loader
transforms_ = [transforms.Resize(image_size, Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

# For test set
dataloader_test = DataLoader(ImageDataset(data_path, transforms_=transforms_, mode='test'),
                             batch_size=1, shuffle=True, num_workers=n_cpu)

###################################

###### Testing######
TESTING_SIZE = 2000

output_path_ = os.path.join(output_imgs_path, str(load_iter))
safe_mkdirs(output_path_)

for j, batch_ in enumerate(dataloader_test):
    if j < TESTING_SIZE:
        real_A_test = Variable(input_A.copy_(batch_['A']))
        real_B_test = Variable(input_B.copy_(batch_['B']))

        fake_AB_test = netG_A2B(real_A_test)
        fake_BA_test = netG_B2A(real_B_test)

        recovered_ABA_test = netG_B2A(fake_AB_test)
        recovered_BAB_test = netG_A2B(fake_BA_test)

        test_product_name_A = batch_['img_A'][0]
        res_A = test_product_name_A.split("/")[-1]
        res_A = res_A.split(".")[0]

        # fn = os.path.join(output_path_, str(j))
        fn_A = os.path.join(output_path_, res_A)
        
        test_product_name_B= batch_['img_B'][0]
        res_B = test_product_name_B.split("/")[-1]
        res_B = res_B.split(".")[0]

        # fn = os.path.join(output_path_, str(j))
        fn_B = os.path.join(output_path_, res_B)
        
        # A_test = np.hstack([tensor2image(real_A_test[0]), tensor2image(fake_AB_test[0]),
        #                     tensor2image(recovered_ABA_test[0])])
        # B_test = np.hstack([tensor2image(real_B_test[0]), tensor2image(fake_BA_test[0]),
        #                     tensor2image(recovered_BAB_test[0])])

        A_test = np.hstack([tensor2image(fake_AB_test[0])])
        B_test = np.hstack([tensor2image(fake_BA_test[0])])

        imageio.imwrite(fn_A + '_A.jpg', A_test)
        imageio.imwrite(fn_B + '_B.jpg', B_test)

    else:
        break