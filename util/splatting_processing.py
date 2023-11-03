import torch
import numpy as np

from motion_diffusion.splatting.splat import splat
from motion_diffusion.splatting.resnet34 import ResNetExtractor

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

class SplattingProcessor():
    def __init__(self, verbose=False, device="cuda"):
        self.resnet_extractor = ResNetExtractor(device)
        self.flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
        self.flow_model.eval()

        self.verbose = verbose
        self.device = device

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    def rescale_flow(self, flow, target_h, target_w):
        """
            Scales flow (spatially and the magnitudes) to the target_w_h.
            Used to scale flow map to different feature scales.
        """
        self.log(f'[SplattingFrameDataset.rescale_flow] rescaling flow from {flow.shape} to {target_h}x{target_w}')
        bs = 1
        if flow.ndim == 4:
            bs, _, h, w = flow.shape
        elif flow.ndim == 3:
            _, h, w = flow.shape
        else:
            raise ValueError(f"Expected flow to be 3 or 4 dimensional, but got {flow.ndim}")

        target_shape = (bs, 2, target_h, target_w)
        scaled_flow = torch.nn.functional.interpolate(flow, size=(target_h, target_w), mode='bilinear', align_corners=False)
        scaled_flow *= torch.tensor([target_h / h, target_w / w], device=scaled_flow.device).reshape(1, 2, 1, 1)

        self.log(f'[SplattingFrameDataset.rescale_flow] rescaled flow to {scaled_flow.shape}')
        return scaled_flow

    def compute_splatted_source_features(self, source_frame, flow):
        # drop first resnet feature
        source_features = self.resnet_extractor(source_frame)
        assert len(source_features) == 5, f"Expected 4 resnet features but got {len(source_features)}"

        # use flow magnitude, scaled as importance metric
        flow_magnitude = torch.norm(flow.clone(), dim=1, keepdim=True)
        flow_magnitude /= torch.max(flow_magnitude)
        # -> (bs, 1, w, h)
        splatted_source_features = []
        scaled_flows = []
        for i, source_feature in enumerate(source_features):
            self.log(f'[compute_splatted_source_features] splatting source feature {i} of shape {source_feature.shape}; flow.shape={flow.shape}')
            source_feature.to(self.device)
            flow.to(self.device)
            flow_magnitude.to(self.device)


            # rescale flow relative to feature size. original flow is at source_frame size
            scaled_flow = self.rescale_flow(flow.clone(), source_feature.shape[-2], source_feature.shape[-1])
            if self.verbose:
                np.save(f'source_feature_{i}.npy', source_feature.cpu().numpy())
                np.save(f'scaled_flow_{i}.npy', scaled_flow.cpu().numpy())
                np.save(f'flow_magnitude_{i}.npy', flow_magnitude.cpu().numpy())
            
            splatted_source_feature = splat(
                features=source_feature.clone(), 
                flow=scaled_flow, 
                importance_metric_Z=flow_magnitude
            )[0]
            splatted_source_features.append(splatted_source_feature)
            scaled_flows.append(scaled_flow)
        
        return splatted_source_features, source_features, flow_magnitude, scaled_flows

    def compute_flow(self, source_frame, target_frame):
        """
            Returns: torch.Size([bs, 2, h, w])
        """
        # todo: move this out of dataset to run batched
        with torch.no_grad():
            flow = self.flow_model(source_frame.to(self.device), target_frame.to(self.device))[-1]
        return flow
        

    def __call__(self, batch):
        """
            Args:
                source_frame: torch.Size([bs, 3, h, w])
                target_frame: torch.Size([bs, 3, h, w])

            Returns: {
                'splatted_source_features': [
                    torch.Size([bs, 65, 56, 56]),
                    torch.Size([bs, 129, 28, 28]),
                    torch.Size([bs, 257, 14, 14]),
                    torch.Size([bs, 513, 7, 7])
                ],
                'flow': torch.Size([bs, 2, 512, 512]),
                
                
                # for debugging
                'source_features': {
                    torch.Size([bs, 64, 56, 56]),
                    torch.Size([bs, 128, 28, 28]),
                    torch.Size([bs, 256, 14, 14]),
                    torch.Size([bs, 512, 7, 7])
                },
                'flow_magnitude': torch.Size([bs, 1, 512, 512]),
                'scaled_flows': [
                    torch.Size([bs, 2, 56, 56]),
                    torch.Size([bs, 2, 28, 28]),
                    torch.Size([bs, 2, 14, 14]),
                    torch.Size([bs, 2, 7, 7])
                ]
            }
        """

        source_frame = batch['source_frame']
        target_frame = batch['target_frame']

        # Compute splatted source features and flow
        flow = self.compute_flow(source_frame, target_frame)
        splatted_source_features, source_features, flow_magnitude, scaled_flows = self.compute_splatted_source_features(source_frame, flow)
        return {
            "splatted_source_features": splatted_source_features,
            "flow": flow.squeeze(0),
            "source_features": source_features,
            "flow_magnitude": flow_magnitude,
            "scaled_flows": scaled_flows,

            "image": batch['source_frame'],
            "target_frame": batch['target_frame'],
            'source_frame': batch['source_frame'],
        }