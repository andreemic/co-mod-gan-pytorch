import pdb
import random
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from models.networks.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from models.networks.stylegan2 import PixelNorm, EqualLinear, EqualConv2d,ConvLayer,StyledConv,ToRGB,ConvToRGB,TransConvLayer
import numpy as np


from PIL import Image

from models.networks.base_network import BaseNetwork

#----------------------------------------------------------------------------
# Mapping network.
# Transforms the input latent code (z) to the disentangled latent code (w).
# Used in configs B-F (Table 1).

import json
def stringify_tensor_object(thing):
    if type(thing) == list:
        return [stringify_tensor_object(x) for x in thing]
    elif type(thing) == dict:
        return json.dumps({key: stringify_tensor_object(value) for key, value in thing.items()}, indent=2)
    elif type(thing) == torch.Tensor:
        return str(thing.shape)
    else:
        return str(thing)
class G_mapping(nn.Module):
    def __init__(self,
            opt
        ):
        latent_size             = 512          # Latent vector (Z) dimensionality.
        label_size              = 0            # Label dimensionality, 0 if no labels.
        dlatent_broadcast       = None         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
        mapping_layers          = 8            # Number of mapping layers.
        mapping_fmaps           = 512          # Number of activations in the mapping layers.
        mapping_lrmul           = 0.01         # Learning rate multiplier for the mapping layers.
        mapping_nonlinearity    = 'lrelu'      # Activation function: 'relu', 'lrelu', etc.
        normalize_latents       = True         # Normalize latent vectors (Z) before feeding them to the mapping layers?
        super().__init__()
        layers = []

        # Embed labels and concatenate them with latents.
        if label_size:
            raise NotImplementedError

        # Normalize latents.
        if normalize_latents:
            layers.append(
                    ('Normalize', PixelNorm()))
        # Mapping layers.
        dim_in = latent_size
        for layer_idx in range(mapping_layers):
            fmaps = opt.dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            layers.append(
                    (
                        'Dense%d' % layer_idx,
                        EqualLinear(
                            dim_in,
                            fmaps,
                            lr_mul=mapping_lrmul,
                            activation="fused_lrelu")
                        ))
            dim_in = fmaps
        # Broadcast.
        if dlatent_broadcast is not None:
            raise NotImplementedError
        self.G_mapping = nn.Sequential(OrderedDict(layers))

    def forward(
            self,
            latents_in):
        styles = self.G_mapping(latents_in)
        return styles

#----------------------------------------------------------------------------
# CoModGAN synthesis network.

"""
block       x_in                            y_in                                resnet feature
Encoder
->
G_4x4       torch.Size([bs, 512, 4, 4])      torch.Size([bs, 3, 4, 4])
G_8x8       torch.Size([bs, 512, 8, 8])      torch.Size([bs, 3, 8, 8])
G_16x16     torch.Size([bs, 512, 16, 16])    torch.Size([bs, 3, 16, 16])
G_32x32     torch.Size([bs, 512, 32, 32])    torch.Size([bs, 3, 32, 32])        (512, 7, 7)
G_64x64     torch.Size([bs, 512, 64, 64])    torch.Size([bs, 3, 64, 64]) <-[]-- (256, 14, 14)
G_128x128   torch.Size([bs, 256, 128, 128])  torch.Size([bs, 3, 128, 128])      (128, 28, 28)
G_256x256   torch.Size([bs, 128, 256, 256])  torch.Size([bs, 3, 256, 256])      (64, 56, 56)
       ->   torch.Size([bs, 64, 512, 512])   torch.Size([bs, 3, 512, 512])

"""
"""
ideas:
    - can implement resnet to deliver quadratic features
"""
# splatted_source_features [
# 0: (bs, 65, 56, 56), 
# 1: (bs, 65, 56, 56), 
# 2: (bs, 129, 28, 28), 
# 3: (bs, 257, 14, 14), 
# 4: (bs, 513, 7, 7)
# ]
resnet_feature_mapping = {
    'G_8x8': 4,
    'G_16x16': 3,
    'G_32x32': 2,
    'G_64x64': 0,
    'G_128x128': 0,
    'G_256x256': 0,
}

class G_synthesis_co_mod_gan(nn.Module):
    def __init__(
            self,
            opt
            ):
        resolution_log2 = int(np.log2(opt.crop_size))
        assert opt.crop_size == 2**resolution_log2 and opt.crop_size >= 4
        def nf(stage): return np.clip(int(opt.fmap_base / (2.0 ** (stage * opt.fmap_decay))), opt.fmap_min, opt.fmap_max)
        assert opt.architecture in ['skip']
        assert opt.nonlinearity == 'lrelu'
        assert opt.fused_modconv
        assert not opt.pix2pix
        self.nf = nf
        super().__init__()
        act = opt.nonlinearity
        self.num_layers = resolution_log2 * 2 - 2
        self.resolution_log2 = resolution_log2
        self.verbose = opt.verbose

        # if False, doesn't feed the raw RGB image as input to the encoder and only infers the target frame using the ResNet features
        self.use_encoder = opt.use_encoder

        log = self.log

        class E_fromrgb(nn.Module): # res = 2..resolution_log2
            def __init__(self, res, channel_in=opt.num_channels):
                super().__init__()
                self.FromRGB = ConvLayer(
                        channel_in,
                        nf(res-1),
                        1,
                        blur_kernel=opt.resample_kernel,
                        activate=True)
            def forward(self, data):
                y, E_features = data
                t = self.FromRGB(y)
                log(self, t, 'E_fromrgb(y)')
                return t, E_features
        class E_block(nn.Module): # res = 2..resolution_log2
            def __init__(self, res):
                super().__init__()
                self.Conv0 = ConvLayer(
                        nf(res-1),
                        nf(res-1),
                        kernel_size=3,
                        activate=True)
                self.Conv1_down = ConvLayer(
                        nf(res-1),
                        nf(res-2),
                        kernel_size=3,
                        downsample=True,
                        blur_kernel=opt.resample_kernel,
                        activate=True)
                self.res = res
            def forward(self, data):
                x, E_features = data
                x = self.Conv0(x)
                E_features[self.res] = x
                x = self.Conv1_down(x)

                log(self, x, 'x after E_block')
                return x, E_features
        class E_block_final(nn.Module): # res = 2..resolution_log2

        
            def __init__(self):
                super().__init__()
                self.Conv = ConvLayer(
                        nf(2),
                        nf(1),
                        kernel_size=3,
                        activate=True)
                self.Dense0 = EqualLinear(nf(1)*4*4, nf(1)*2,
                                activation="fused_lrelu")
                self.dropout = nn.Dropout(opt.dropout_rate)
            def forward(self, data):
                x, E_features = data
                log(self, x, 'x before E_block_final')
                x = self.Conv(x)
                E_features[2] = x
                bsize = x.size(0)
                x = x.view(bsize, -1)

                log(self, x, f'x in E_block_final, before Dense0={self.Dense0}')
                x = self.Dense0(x)
                x = self.dropout(x)
                return x, E_features
        # Custom: in original co-mod-gan this is channel_in=opt.num_channels+1; here we don't need the mask channel
        def make_encoder(channel_in=opt.num_channels):
            Es = []
            for res in range(self.resolution_log2, 2, -1):
                if res == self.resolution_log2:
                    Es.append(
                            (
                                '%dx%d_0' % (2**res, 2**res),
                                E_fromrgb(res, channel_in)
                                ))
                Es.append(
                        (
                            '%dx%d' % (2**res, 2**res),
                            E_block(res)

                            ))
            # Final layers.
            Es.append(
                    (
                        '4x4',
                        E_block_final()

                        ))
            Es = nn.Sequential(OrderedDict(Es))
            return Es
        self.make_encoder = make_encoder

        # Main layers.
        # Custom: in original co-mod-gan this is opt.num_channels+1; here we don't need the mask channel
        c_in = opt.num_channels
        if self.use_encoder:
            self.E = self.make_encoder(channel_in=c_in)
        else:
            self.E = None
            print(f'use_encoder=False -> not using an encoder in the generator')

        # Single convolution layer with all the bells and whistles.
        # Building blocks for main layers.
        mod_size = 0
        if opt.style_mod:
            mod_size += opt.dlatent_size
        if opt.cond_mod:
            mod_size += nf(1)*2
        assert mod_size > 0
        self.mod_size = mod_size
        def get_mod(latent, idx, x_global):
            if isinstance(latent, list):
                latent = latent[:][idx]
            else:
                latent = latent[:,idx]
            mod_vector = []
            if opt.style_mod:
                mod_vector.append(latent)
            if opt.cond_mod:
                mod_vector.append(x_global)
            mod_vector = torch.cat(mod_vector, 1)
            return mod_vector
        self.get_mod = get_mod
        class Block(nn.Module):
            def __init__(self, res):
                super().__init__()
                self.res = res
                self.resnet_e
                self.Conv0_up = StyledConv(
                        nf(res-2),
                        nf(res-1),
                        kernel_size=3,
                        style_dim=mod_size,
                        upsample=True,
                        blur_kernel=opt.resample_kernel)
                self.Conv1 = StyledConv(
                        nf(res-1),
                        nf(res-1),
                        kernel_size=3,
                        style_dim=mod_size,
                        upsample=False)
                self.ToRGB = ToRGB(
                        nf(res-1),
                        mod_size, out_channel=opt.num_channels)
            def forward(self, x, y, dlatents_in, x_global, E_features, resnet_feature):
                x_skip = E_features[self.res]
                
                mod_vector = get_mod(dlatents_in, res*2-5, x_global)
                if opt.noise_injection:
                    noise = None
                else:
                    noise = 0

                # Warning: the x_skip is only used in WeightedConv2d layers but here we are not using them. 
                # So x_skip only gets added here as residual connection but is not used in the convolutional blocks.
                x = self.Conv0_up(x, mod_vector, noise, x_skip=x_skip)
                x = x + x_skip
                mod_vector = get_mod(dlatents_in, self.res*2-4, x_global)
                x = self.Conv1(x, mod_vector, noise, x_skip=x_skip)
                mod_vector = get_mod(dlatents_in, self.res*2-3, x_global)
                y = self.ToRGB(x, mod_vector, skip=y, x_skip=x_skip)
                return x, y
        self.Block = Block
        class Block0(nn.Module):
            def __init__(self):
                super().__init__()
                self.Dense = EqualLinear(
                        nf(1)*2,
                        nf(1)*4*4,
                        activation="fused_lrelu")
                self.Conv = StyledConv(
                        nf(1),
                        nf(1),
                        kernel_size=3,
                        style_dim=mod_size,
                        )
                self.ToRGB = ToRGB(
                        nf(1),
                        style_dim=mod_size,
                        upsample=False, out_channel=opt.num_channels)
            def forward(self, x, dlatents_in, x_global):
                x = self.Dense(x)
                x = x.view(-1, nf(1), 4, 4)
                mod_vector = get_mod(dlatents_in, 0, x_global)
                if opt.noise_injection:
                    noise = None
                else:
                    noise = 0
                x = self.Conv(x, mod_vector, noise)
                mod_vector = get_mod(dlatents_in, 1, x_global)
                y = self.ToRGB(x, mod_vector)
                return x, y
        # Early layers.
        self.G_4x4 = Block0()
        # Main layers.
        for res in range(3, resolution_log2 + 1):
            setattr(self, 'G_%dx%d' % (2**res, 2**res),
                    Block(res))

    def log(self, module, tensor, label=''):
        if self.verbose:
            print(f'{self.__class__.__name__} -> {module.__class__.__name__} {label} {stringify_tensor_object(tensor)}')

    def forward(self, source_frame, splatted_source_features, dlatents_in):
        """.git/

        Args:
            - source_frame: torch.Size([bs, 3, h, w])
            - splatted_source_features: list of torch.Size([bs, ch, h, w]) (resnet features on different scales)
            - dlatents_in: torch.Size([bs, 14, 512])
        """
        # masks are in [0; 1]; map them to [-0.5; 0.5]
        # apply masks to images
        if not self.use_encoder:
            raise NotImplementedError('Generator forward pass use_encoder=False not implemented yet.')
        y = source_frame
        E_features = {}
        x_global, E_features = self.E((y, E_features))

        self.log(self, x_global, '\n\n --- Encoder Output (x_global):')
        self.log(self, E_features, '\n\n --- Encoder Output (E_features):')


        # -> x_global: torch.Size([bs, 1024])
        """ 
                E_features: {
                    9: "torch.Size([1, 64, 512, 512])",
                    8: "torch.Size([1, 128, 256, 256])",
                    7: "torch.Size([1, 256, 128, 128])",
                    6: "torch.Size([1, 512, 64, 64])",
                    5: "torch.Size([1, 512, 32, 32])",
                    4: "torch.Size([1, 512, 16, 16])",
                    3: "torch.Size([1, 512, 8, 8])",
                    2: "torch.Size([1, 512, 4, 4])"
                }
        """

        # toprint: dimensions of x_global and length of E_features
        x = x_global
        x, y = self.G_4x4(x, dlatents_in, x_global)
        np.save('y_4x4', y.clone().detach().cpu().numpy())
        for res in range(3, self.resolution_log2 + 1):
            block_str = 'G_%dx%d' % (2**res, 2**res)
            block = getattr(self, block_str)
            self.log(block, x, f'x before block {block_str}')
            self.log(block, y, f'y before block {block_str}')
            print('\n')

            resnet_feature = splatted_source_features[resnet_feature_mapping[block_str]] if block_str in resnet_feature_mapping else None
            self.log(block, resnet_feature, f'resnet_feature for block {block_str}')
            x, y = block(x, y, dlatents_in, x_global, E_features, resnet_feature)
            

            # log png intermediate images
            # y_str = 'y_%dx%d' % (2**res, 2**res)
            # Image.fromarray(((y[0].clone().detach().cpu().numpy() / 2 + 0.5) * 255).astype('uint8').transpose(2, 1, 0)).save(f'{y_str}.png')

        self.log(self, y, 'y after all blocks')
        raw_out = y
        images_out = y 
        return images_out, raw_out

#----------------------------------------------------------------------------
# Main generator network.
# Composed of two sub-networks (mapping and synthesis) that are defined below.
# Used in configs B-F (Table 1).

class Generator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--dlatent_size',      type=int, default= 512         )# Disentangled latent (W) dimensionality.
        parser.add_argument('--num_channels',      type=int, default= 3,            )# Number of output color channels.
        parser.add_argument('--fmap_base',         type=int, default= 16 << 10,     )# Overall multiplier for the number of feature maps.
        parser.add_argument('--fmap_decay',        type=int, default= 1.0,          )# log2 feature map reduction when doubling the resolution.
        parser.add_argument('--fmap_min',          type=int, default= 1,            )# Minimum number of feature maps in any layer.
        parser.add_argument('--fmap_max',          type=int, default= 512,          )# Maximum number of feature maps in any layer.
        parser.add_argument('--randomize_noise',   type=bool, default= True,         )# True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        parser.add_argument('--architecture',      type=str, default= 'skip',       )# Architecture: 'orig', 'skip', 'resnet'.
        parser.add_argument('--nonlinearity',      type=str, default= 'lrelu',      )# Activation function: 'relu', 'lrelu', etc.
        parser.add_argument('--resample_kernel',   type=list, default= [1,3,3,1],    )# Low-pass filter to apply when resampling activations. None = no filtering.
        parser.add_argument('--fused_modconv',     type=bool, default= True,         )# Implement modulated_conv2d_layer() as a single fused op?
        parser.add_argument('--pix2pix',           type=bool, default= False)
        parser.add_argument('--dropout_rate',      type=float, default= 0.5)
        parser.add_argument('--cond_mod',          type=bool, default= True,)
        parser.add_argument('--style_mod',         type=bool, default= True,)
        parser.add_argument('--noise_injection',   type=bool, default= True,)
        return parser
    def __init__(
            self,
            opt=None):                                          # Arguments for sub-networks (mapping and synthesis).
        super().__init__()
        self.G_mapping = G_mapping(opt)
        self.G_synthesis = G_synthesis_co_mod_gan(opt)

    def forward(
            self,
            images_in=None,
            masks_in=None,
            latents_in=None,
            return_latents=False,
            inject_index=None,
            truncation=None,
            truncation_latent=None,
            input_is_latent=False,
            get_latent=False,
            ):

        """
        Args:
            - images_in: torch.Size([bs, 3, h, w]) \in [-1, 1]
            - masks_in: torch.Size([bs, 1, h, w]) \in [0, 1]
            - latents_in: torch.Size([bs, 512]) \in [-1, 1]
        """
        #assert isinstance(latents_in, list)
        if not input_is_latent:
            dlatents_in = [self.G_mapping(s) for s in latents_in]
            # -> dlatents_in: list of torch.Size([bs, 512])
        else:
            dlatents_in = latents_in
        if get_latent:
            return dlatents_in
        if truncation is not None:
            dlatents_t = []
            for style in dlatents_in:
                dlatents_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            dlatents_in = dlatents_t

        
        if len(dlatents_in) < 2:
            inject_index = self.G_synthesis.num_layers # (16)
            if dlatents_in[0].ndim < 3:
                # ? if batch size == 1 and latents like [bs, 512], multiply them 16-wise into [bs, 16, 512]
                dlatent = dlatents_in[0].unsqueeze(1).repeat(1, inject_index, 1)
                # -> dlatent = torch.Size([bs, 16, 512])
            else:
                # ? if batch size == 1 and latents like [bs, 16, 512], just take the first one
                # ... not sure why.. wouldn't we end up with [16, 512] dlatent in this case? maybe some special input format
                dlatent = dlatents_in[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.G_synthesis.num_layers - 1)
            dlatent = dlatents_in[0].unsqueeze(1).repeat(1, inject_index, 1)
            dlatent2 = dlatents_in[1].unsqueeze(1).repeat(1, self.G_synthesis.num_layers - inject_index, 1)

            dlatent = torch.cat([dlatent, dlatent2], 1)
        output, raw_out = self.G_synthesis(images_in, masks_in, dlatent)
        if return_latents:
            return output, raw_out, dlatent
        else:
            return output, raw_out, None

#----------------------------------------------------------------------------
# CoModGAN discriminator.

class Discriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--mbstd_num_features',      type=int, default= 1,          )# Number of features for the minibatch standard deviation layer.
        parser.add_argument('--mbstd_group_size',      type=int, default= 4,            )# Group size for the minibatch standard deviation layer, 0 = disable.
        return parser
    def __init__(
            self,
            opt):
        label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        architecture        = 'resnet'     # Architecture: 'orig', 'skip', 'resnet'.
        pix2pix             = False
        assert not pix2pix
        assert opt.nonlinearity == 'lrelu'
        assert architecture == 'resnet'
        if opt is not None:
            resolution = opt.crop_size

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        def nf(stage): return np.clip(int(opt.fmap_base / (2.0 ** (stage * opt.fmap_decay))), opt.fmap_min, opt.fmap_max)
        #assert architecture in ['orig', 'skip', 'resnet']

        # Building blocks for main layers.
        super().__init__()
        layers = []
        c_in = opt.num_channels
        layers.append(
                (
                    "ToRGB",
                    ConvLayer(
                        c_in,
                        nf(resolution_log2-1),
                        kernel_size=3,
                        activate=True)
                    )
                )

        class Block(nn.Module):
            def __init__(self, res):
                super().__init__()
                self.Conv0 = ConvLayer(
                        nf(res-1),
                        nf(res-1),
                        kernel_size=3,
                        activate=True)
                self.Conv1_down = ConvLayer(
                        nf(res-1),
                        nf(res-2),
                        kernel_size=3,
                        downsample=True,
                        blur_kernel=opt.resample_kernel,
                        activate=True)
                self.Skip = ConvLayer(
                        nf(res-1),
                        nf(res-2),
                        kernel_size=1,
                        downsample=True,
                        blur_kernel=opt.resample_kernel,
                        activate=False,
                        bias=False)
            def forward(self, x):
                t = x
                x = self.Conv0(x)
                x = self.Conv1_down(x)
                t = self.Skip(t)
                x = (x + t) * (1/np.sqrt(2))
                return x
        # Main layers.
        for res in range(resolution_log2, 2, -1):
            layers.append(
                    (
                        '%dx%d' % (2**res, 2**res),
                        Block(res)
                        )
                    )
        self.convs = nn.Sequential(OrderedDict(layers))
        # Final layers.
        self.mbstd_group_size = opt.mbstd_group_size
        self.mbstd_num_features = opt.mbstd_num_features

        self.Conv4x4 = ConvLayer(nf(1)+1, nf(1), kernel_size=3, activate=True)
        self.Dense0 = EqualLinear(nf(1)*4*4, nf(0), activation='fused_lrelu')
        self.Output = EqualLinear(nf(0), 1)

    def forward(self, images_in, masks_in):
        masks_in = 1-masks_in
        y = torch.cat([masks_in - 0.5, images_in], 1)
        out = self.convs(y)
        batch, channel, height, width = out.shape
        group_size = min(batch, self.mbstd_group_size)
        #print(out.shape)
        #pdb.set_trace()
        stddev = out.view(
            group_size,
            -1,
            self.mbstd_num_features,
            channel // self.mbstd_num_features,
            height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group_size, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out = self.Conv4x4(out)
        out = out.view(batch, -1)
        out = self.Dense0(out)
        out = self.Output(out)
        return out



if __name__ == "__main__":
    import cv2
    from PIL import Image
    path_img = "/home/zeng/co-mod-gan/imgs/example_image.jpg"
    path_mask = "/home/zeng/co-mod-gan/imgs/example_mask.jpg"

    real = np.asarray(Image.open(path_img)).transpose([2, 0, 1])/255.0

    masks = np.asarray(Image.open(path_mask).convert('1'), dtype=np.float32)

    images = torch.Tensor(real.copy())[None,...]*2-1
    masks = torch.Tensor(masks)[None,None,...].float()
    masks = (masks==0).float()

    net = Discriminator()
    hh = net(images, masks)
    pdb.set_trace()

    #net = Generator()
    #net.G_mapping.load_from_tf_dict("/home/zeng/co-mod-gan/co-mod-gan-ffhq-9-025000.npz")
    #net.G_synthesis.load_from_tf_dict("/home/zeng/co-mod-gan/co-mod-gan-ffhq-9-025000.npz")
    #net.eval()
    #torch.save(net.state_dict(), "co-mod-gan-ffhq-9-025000.pth")

    #latents_in = torch.randn(1, 512)

    #hh = net(images, masks, [latents_in], truncation=None)
    #hh = hh.detach().cpu().numpy()
    #hh = (hh+1)/2
    #hh = (hh[0].transpose((1,2,0)))*255
    #cv2.imwrite("hh.png", hh[:,:,::-1].clip(0,255))
    #pdb.set_trace()
