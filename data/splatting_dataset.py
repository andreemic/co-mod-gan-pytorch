import torch
from torch.utils.data import Dataset
import random
import numpy as np

from data.base_dataset import get_params, get_transform, BaseDataset

import sys
import os 
MOTION_DIFFUSION_PATH = os.environ.get("MOTION_DIFFUSION_PATH", '../../../../')

sys.path.append(MOTION_DIFFUSION_PATH)
sys.path.append("/export/compvis-nfs/user/mandreev/models/gan/co-mod-gan-pytorch/data")
sys.path.append("/export/compvis-nfs/user/mandreev/kim/motion_diffusion/")

from motion_diffusion.data.video import VideoDataset


class SplattingFrameDataset(Dataset):
    def __init__(self, video_dataset_kwargs, num_samples_per_sequence=4):
        """
        Initializes the dataset.

        Args:
        video_dataset (Dataset): An instance of VideoDataset.
        min_frame_interval (int): Minimum interval between selected frames.
        max_frame_interval (int): Maximum interval between selected frames.
        """

        
        self.video_dataset = VideoDataset(**video_dataset_kwargs)
        assert self.video_dataset.single, "VideoDataset must be in single-mode"
        self.num_samples_per_sequence = num_samples_per_sequence

    def initialize(self, opt):
        print(f'Initializing dataset {opt.dataset_mode} ...')
        self.opt = opt
        pass

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.video_dataset) * self.num_samples_per_sequence

    def __getitem__(self, idx):
        """
        Returns a data sample from the dataset.

        Args:
        idx (int): Index of the data sample.

        Returns:
            {
                'source_frame': torch.Size([3, h, w]),
                'target_frame': torch.Size([3, h, w]),
            }
        """
    
        # Get the sample definition
        sequence_idx = idx // self.num_samples_per_sequence

        video_data = self.video_dataset[sequence_idx]['sequence']

        # Select source and target frames based on the random interval
        source_frame = video_data[:, 0, :, :]
        target_frame = video_data[:, 1, :, :]
    
        
        # image_size = (source_frame.size(-1), source_frame.size(-2))
        # params = get_params(self.opt, image_size)
        # transform_image = get_transform(self.opt, params)

        # def apply_transforms(image_pt):
        #     image_pil = Image.fromarray((255 * (image_pt.numpy() + 1) / 2).astype('uint8'))
        #     return transform_image(image_pil)

        return {
            "source_frame": (source_frame),
            "target_frame": (target_frame),
        }


# for comod gan
class SplattingDataset(BaseDataset):

    def initialize(self, opt):
        self.ds = SplattingFrameDataset(video_dataset_kwargs=dict(
            data_path='/export/compvis-nfs/group/datasets/taichi/',
            typ="taichi",
            sequence_length=16,
            context_lengt=0,
            frameskip=0,
            size=256,
            single=True,
            single_min_skip=1,
            single_max_skip=3,
            split="train",
            use_action_window=True,
            taichi_raw=False
        ))

        self.ds.initialize(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]

    
