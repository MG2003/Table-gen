import torch
#Constants

latent_z = 200
generator_lr = 0.0025
d_lr = 0.00001
batch_size = 100
dimension = 32


class Generator(torch.nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.resolution = args.resolution
        self.root = args.root

        padding = (1, 1, 1)

        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(latent_z, dimension * 8, kernel_size = 4, stride = 2, bias= args.bias, paddding=padding),
            torch.nn.BatchNorm3d(dimension * 8),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose3d(dimension*8, dimension*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(dimension*4),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose3d(dimension*4, dimension*2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(dimension*2),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose3d(dimension*2, dimension, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(dimension),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose3d(dimension, 1, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )
    def forward(self, input):
        output = input.view(-1, latent_z, 1, 1, 1)   
        return self.layers(output) 



class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.resolution = args.resolution
        self.root = args.root

        padding = (1, 1, 1)

        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(latent_z, dimension * 8, kernel_size = 4, stride = 2, bias= args.bias, paddding=padding),
            torch.nn.BatchNorm3d(dimension * 8),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose3d(dimension*8, dimension*4, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(dimension*4),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose3d(dimension*4, dimension*2, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(dimension*2),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose3d(dimension*2, dimension, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(dimension),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose3d(dimension, 1, kernel_size=4, stride=2, bias=args.bias, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )
    def forward(self, input):
        output = input.view(-1, 1, dimension, dimension, dimension)   
        return self.layers(output) 
         