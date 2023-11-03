import sys
import numpy as np
import os 
import torch
MOTION_DIFFUSION_PATH = os.environ.get("MOTION_DIFFUSION_PATH", '/export/home/mandreev/kim/motion_diffusion')

sys.path.append(MOTION_DIFFUSION_PATH)
sys.path.append("/export/compvis-nfs/user/mandreev/models/gan/co-mod-gan-pytorch/data")
sys.path.append("/export/compvis-nfs/user/mandreev/kim/motion_diffusion/")

from splatting_processing import SplattingProcessor

from PIL import Image
frame1 = Image.open('/export/home/mandreev/tracking/omnimotion/omnimotion_videos/soccer_short/color/00001.jpg')
frame2 = Image.open('/export/home/mandreev/tracking/omnimotion/omnimotion_videos/soccer_short/color/00015.jpg')

batch_size = 4
h = 256
w = 256

source_frame = torch.tensor(np.array(frame1.resize((h, w))).astype('float32').transpose(2, 0, 1) / 127.5 - 1).cuda().unsqueeze(0).repeat(batch_size, 1, 1, 1)
target_frame = torch.tensor(np.array(frame2.resize((h, w))).astype('float32').transpose(2, 0, 1) / 127.5 - 1).cuda().unsqueeze(0).repeat(batch_size, 1, 1, 1)


splatting_processor = SplattingProcessor(verbose=True, device="cuda")

result = splatting_processor(source_frame, target_frame)

def stringify_tensor_object(thing):
    if type(thing) == list:
        return [stringify_tensor_object(x) for x in thing]
    elif type(thing) == dict:
        return {key: stringify_tensor_object(value) for key, value in thing.items()}
    elif type(thing) == torch.Tensor:
        return str(thing.shape)
    else:
        return str(thing)

import json
print(f'\n\n---- Output of splatting_processor.__call__ ----')
for key, value in result.items():
    print(f'{key}: {json.dumps(stringify_tensor_object(value), indent=2)}')

assert result['source_features'][0].shape == (batch_size, 64, 56, 56)
assert result['source_features'][1].shape == (batch_size, 64, 56, 56)
assert result['source_features'][2].shape == (batch_size, 128, 28, 28)
assert result['source_features'][3].shape == (batch_size, 256, 14, 14)
assert result['source_features'][4].shape == (batch_size, 512, 7, 7)

assert result['splatted_source_features'][0].shape == (batch_size, 65, 56, 56)
assert result['splatted_source_features'][1].shape == (batch_size, 65, 56, 56)
assert result['splatted_source_features'][2].shape == (batch_size, 129, 28, 28)
assert result['splatted_source_features'][3].shape == (batch_size, 257, 14, 14)
assert result['splatted_source_features'][4].shape == (batch_size, 513, 7, 7)

assert result['flow'].shape == (batch_size, 2, h, w)
assert result['flow_magnitude'].shape == (batch_size, 1, h, w)

print(f'All shape checks passed ✅ ')





# log Resnet Features
from motion_diffusion.splatting.resnet34.vis import visualize_resnet_features
import os
outdir = './splatting_processing_test_images'
os.makedirs(outdir, exist_ok=True)
visualize_resnet_features(result['source_features'][0][:1], return_pil=True).save(os.path.join(outdir, 'source_feature_0.png'))
visualize_resnet_features(result['splatted_source_features'][0][:1], return_pil=True).save(os.path.join(outdir, 'splatted_source_feature_0.png'))


import motion_diffusion.splatting.resnet34 as resnet34
resnet_extractor = resnet34.ResNetExtractor()
directly_extracted_ft = resnet_extractor(frame1, verbose=True)
visualize_resnet_features(directly_extracted_ft[0], return_pil=True).save(os.path.join(outdir, 'directly_extracted_ft_0.png'))
print(f'Saved resnet features to {outdir}')


# log flow
from torchvision.utils import flow_to_image
def tensor_to_pil(x):
    if x.ndim == 4:
        x = x[0]
    if x.device != 'cpu':
        x = x.cpu()
    x = x.numpy()

    if x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    if x.min() < 0.5 and x.max() > 0.5:
        x = (x + 1) / 2

    if x.max() < 5:
        x = x * 255

    return Image.fromarray(x.astype('uint8'))

tensor_to_pil(flow_to_image(result['flow'][0])).save(os.path.join(outdir, 'flow.png'))
print(f'Saved flow to {outdir}')

# log flow magnitude
tensor_to_pil(result['flow_magnitude'][0].squeeze(0)).save(os.path.join(outdir, 'flow_magnitude.png'))