import random

import numpy as np
import torch
from PIL import Image
import os


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
    
def save_config(file_path,**kwargs):

    f_path = os.path.join(file_path, 'configurations.txt')
    

    if not os.path.exists(f_path):
        open(f_path, 'w').close()
    with open(f_path, 'w') as f:
        f.write('Configurations:\n')
        f.write('-' * 70 + '\n')
        f.write('|%25s | %40s|\n' % ('keys', 'values'))
        f.write('-' * 70 + '\n')
        for key, value in kwargs.items():
            f.write('|%25s | %40s|\n' % (str(key), str(value)))
        f.write('-' * 70 + '\n')

def download_weights(backbone, model_dir="./model_data"):

    import os
    from torch.hub import load_state_dict_from_url
    
    # 定义下载链接
    download_urls = {
        # VGG
        'vgg'       : 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg16'       : 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
        
        # resnet
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50'  : 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101' : 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        
        # SqueezeNet
        'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
        'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
        
        # DenseNet
        'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
        'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
        'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',

        # FCN resnet
        'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
        'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
        
        # Faster resnet
        'fasterrcnn_resnet50_fpn_coco':'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        
        # DeepLabV3 ResNet
        'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
        'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',


        'fasterrcnn_resnet50_fpn_coco':'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        

        'maskrcnn_resnet50_fpn_coco':'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',


        'keypointrcnn_resnet50_fpn_coco_legacy':'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-9f466800.pth',
        'keypointrcnn_resnet50_fpn_coco':'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth',
        
        'swin':'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth'

    }
    urls_name = {
        'vgg'       : 'vgg16-397923af.pth',
        'vgg16'     : 'vgg16-397923af.pth',
        'vgg19'     : 'vgg19-dcbb9e9d.pth',
        'vgg16_bn'  : 'vgg16_bn-6c64b313.pth',
        'vgg19_bn'  : 'vgg19_bn-c79401a0.pth',
        'resnet18'  : 'resnet18-5c106cde.pth',
        'resnet34'  : 'resnet34-333f7ec4.pth',
        'resnet50'  : 'resnet50-19c8e357.pth',
        'resnet101' : 'resnet101-5d3b4d8f.pth',
        'resnet152' : 'resnet152-b121ed2d.pth',
        'swin'      : 'swin_tiny_patch4_window7_224_22k.pth'
    }

    url = download_urls[backbone]
    url_name = urls_name[backbone]
    

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_path = os.path.join(model_dir,url_name)
    if os.path.exists(file_path):
        print('已存在预训练权重文件,跳过下载')
    else:
        load_state_dict_from_url(url, model_dir)