import torch
from thop import profile

from nets.MFDA_UNet import Unet


name = 'coco'
if name == 'coco':
    num_classes = 21
    img_size = 256
    img_size = [img_size,img_size]
    backbone='efficientnet-b5'

elif name == 'syn':
    num_classes = 8
    img_size = 224
    img_size = [img_size,img_size]
    backbone='efficientnet-b5'


model = Unet(
    num_classes=num_classes,
    backbone=backbone,  
    pretrained=True,
    is_print_size=True,
)

input_tensor = torch.randn(1, 3 , img_size[0], img_size[1])  # (batch, channel, height, width)


flops, params = profile(model, inputs=(input_tensor,), verbose=False)

print(f"Parameters: {params/1e6:.2f}M")
print(f"FLOPs: {flops/1e9:.2f}G")