#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pickle
import sys

import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 200


# Number of channels in the training images
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 128


class Generator(nn.Module):
    '''
    Defines the generator network architecture for a GAN.

    The sequential module takes a latent vector (Z) as input and progressively 
    upsamples it through a series of transposed convolutional layers to generate 
    an image output. Each upsampling stage increases the spatial resolution while 
    reducing the number of feature maps, following the standard DCGAN pattern.
    '''
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),      # noise vector to 4 by 4 image
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 4, 3, 1, 1, bias=False), # refines the image without upsampling
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False), # 8 by 8
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 2, ngf*2, 3, 1, 1, bias=False),   # refining
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),     # 16 by 16
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, ngf, 3, 1, 1, bias=False),         # refining
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),          # 32 by 32
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    '''
    Defines the discriminator network for a Generative Adversarial Network (GAN).

    It progressively downsamples the input image through a series of convolutional 
    layers, reducing spatial dimensions while increasing feature depth to extract 
    high-level representations useful for classification.
    '''
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            # kernel size=4, stride=2 and padding=1
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),    # 16 by 16     
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), # 8 by 8  
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),   # 4 by 4
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False),   # scalar
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DCGAN")

    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs to train',
        required=True
    )

    parser.add_argument(
        '--checkpoint',
        type=int,
        help='Checkpoint epoch to resume from'
    )

    args = parser.parse_args()
        
    num_epochs=args.epochs
    print(f"Number of epochs: {num_epochs}")
    checkpoint_epoch=None
    try:
        checkpoint_epoch=args.checkpoint
        print(f"checkpoint epoch: {checkpoint_epoch}")
    except:
        pass





    # Set random seed for reproducibility
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    workers = 2
    batch_size = 64
    image_size = 32
    num_eval_images = 10000  # Total images needed for a reliable score
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    dataloader_generator = torch.Generator()


    G_losses, D_losses, img_list = [], [], []
    iters, start_epoch = 0, 0
    fixed_eval_noise=None
    fixed_visualisation_noise=None

    if checkpoint_epoch:
        print("-----------Using saved model---------")
        checkpoint_path = f"checkpoint_epoch_{checkpoint_epoch}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Restore everything from the checkpoint
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        G_losses = checkpoint['G_losses']
        D_losses = checkpoint['D_losses']
        img_list = checkpoint['img_list']
        iters = checkpoint['iters']
        start_epoch = checkpoint['epoch'] + 1
        img_list = [img.cpu() for img in checkpoint['img_list']]

        
        
        # Restore RNG state IMMEDIATELY
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())
        dataloader_generator.set_state(checkpoint['dataloader_rng_state'].cpu())

        
        fixed_visualisation_noise = checkpoint['fixed_visualisation_noise'] # for plotting  
        fixed_eval_noise=checkpoint['fixed_eval_noise'] # for inception score calculation

    
    # Create fixed_noise only if not resuming from a checkpoint
    if not checkpoint_epoch:
        fixed_visualisation_noise = torch.randn(64, nz, 1, 1, device=device)
        fixed_eval_noise=torch.randn(num_eval_images, nz, 1, 1, device=device)
        # Apply weights_init only when training from scratch
        netG.apply(weights_init)
        netD.apply(weights_init)
        dataloader_generator.manual_seed(manualSeed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1,1]
    ])

    dataset = dset.CIFAR10(root="./data", train=True, download=True, transform=transform)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers,generator=dataloader_generator)

    '''
    
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    '''
    

    criterion = nn.BCELoss()


    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    epoch_cut_off=0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(start_epoch,num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            # get a batch of real images
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            # calculate loss for real images
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            # calculate loss for fake images
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            '''
            We use non saturating loss for the generator since it avoids vanishing gradients when discriminator
            becomes too strong. So labels passed to the generator will be 1 instead of 0.
            '''
            label.fill_(real_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate generator loss
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_visualisation_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            

            iters += 1
        
        if((epoch+1)%5==0 and (epoch+1)>=epoch_cut_off):
            checkpoint = {
            'epoch': epoch,
            'iters': iters,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses,
            'img_list': img_list,
            'fixed_visualisation_noise': fixed_visualisation_noise,
            'fixed_eval_noise':fixed_eval_noise,
            'rng_state': torch.get_rng_state(), # Save CPU RNG state
            'cuda_rng_state': torch.cuda.get_rng_state(), # Save current GPU RNG state
            'dataloader_rng_state': dataloader_generator.get_state()
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')

            print(f"--- Generating a fixed set of images for evaluation at epoch {epoch+1} ---")            
            # List to hold the generated batches
            eval_images = []
            eval_batch_size = 64
            
            # Set model to evaluation mode
            netG.eval()
            
            with torch.no_grad():
                for i in range(0, num_eval_images, eval_batch_size):
                    noise_batch = fixed_eval_noise[i:i+eval_batch_size]
                    fake_batch = netG(noise_batch.to(device)).detach().cpu()
                    eval_images.append(fake_batch)
                    

            # Combine all the generated batches into a single tensor
            all_fake_images_tensor = torch.cat(eval_images, dim=0)
            torch.save(all_fake_images_tensor, f"all_fake_images_epoch_{epoch+1}.pt")


            
            # Set model back to training mode
            netG.train()
                
    

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i.cpu(),(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())
    ani.save('training_animation.mp4', writer='ffmpeg', fps=1)
    ani.save('training_animation.gif', writer='pillow', fps=1)

    
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.savefig("fake_images_last_epoch.png", bbox_inches='tight')
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig("fake_images_last_epoch.png", bbox_inches='tight')
    plt.show()
    


