import gradio as gr
import cv2
import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from basicsr.archs.uformer_arch import Uformer
import argparse
from torch.distributions import Normal
import torchvision.transforms as transforms
import os
from basicsr.utils.flare_util import blend_light_source,get_args_from_json,save_args_to_json,mkdir,predict_flare_from_6_channel,predict_flare_from_3_channel

parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str,default='Uformer')
parser.add_argument('--model_path',type=str,default='checkpoint/Uformer_7kpp.pth')
parser.add_argument('--output_ch',type=int,default=6)
parser.add_argument('--flare7kpp', action='store_const', const=True, default=True) #use flare7kpp's inference method and output the light source directly.

args = parser.parse_args()

def load_params(model_path):
     full_model=torch.load(model_path)
     if 'params_ema' in full_model:
          return full_model['params_ema']
     elif 'params' in full_model:
          return full_model['params']
     else:
          return full_model

def remove_flare(image):
    # print(image)
    image = Image.fromarray(image)
    to_tensor=transforms.ToTensor()
    resize=transforms.Resize((512,512))
    merge_img = resize(to_tensor(image))
    merge_img = merge_img.cuda().unsqueeze(0)
    with torch.no_grad():
        output_img=model(merge_img)
        gamma=torch.Tensor([2.2])
        deflare_img,_,_=predict_flare_from_6_channel(output_img,gamma)
    torchvision.utils.save_image(deflare_img, './a.jpg')
    ans = Image.open('./a.jpg').convert('RGB')
    return ans
torch.cuda.empty_cache()
model=Uformer(img_size=512,img_ch=3,output_ch=6).cuda()
model.load_state_dict(load_params(args.model_path))
model.eval()
interface = gr.Interface(fn=remove_flare, inputs="image", outputs="image", examples=[
    ["dataset/test_images/input_000005.png"],
    ["dataset/test_images/input_000017.png"],
    ["dataset/test_images/input_000031.png"],
    ["dataset/test_images/input_000035.png"],
    ["dataset/test_images/input_000052.png"]
])
# interface = gr.Interface(fn=remove_flare, inputs="image", outputs="image")
interface.launch()
