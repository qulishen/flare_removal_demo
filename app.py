import gradio as gr
import cv2
import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageChops
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
class ImageProcessor:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resize_image(self, image, target_size):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width < original_height:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        return image.resize((new_width, new_height))

    def process_image(self, image):
        # Open the original image
        to_tensor=transforms.ToTensor()
        original_image = image

        # Resize the image proportionally to make the shorter side 512 pixels
        resized_image = self.resize_image(original_image, 512)
        resized_width, resized_height = resized_image.size

        # Process each 512-pixel segment separately
        segments = []
        overlaps = []
        if resized_width > 512:
            for end_x in range(512, resized_width+256, 256):
                end_x = min(end_x, resized_width)
                overlaps.append(end_x)
                cropped_image = resized_image.crop((end_x-512, 0, end_x, 512))
                processed_segment = self.model(to_tensor(cropped_image).unsqueeze(0).to(self.device)).squeeze(0)
                segments.append(processed_segment)
        else:
            for end_y in range(512, resized_height+256, 256):
                end_y = min(end_y, resized_height)
                overlaps.append(end_y)
                cropped_image = resized_image.crop((0, end_y-512, 512, end_y))
                processed_segment = self.model(to_tensor(cropped_image).unsqueeze(0).to(self.device)).squeeze(0)
                segments.append(processed_segment)
        overlaps = [0] + [prev - cur + 512 for prev, cur in zip(overlaps[:-1], overlaps[1:])]

        # Blending the segments
        for i in range(1, len(segments)):
            overlap = overlaps[i]
            alpha = torch.linspace(0, 1, steps=overlap).to(self.device)
            if resized_width > 512:
                alpha = alpha.view(1, -1, 1).expand(512, -1, 6).permute(2,0,1)
                segments[i][:, :, :overlap] = alpha * segments[i][:, :, :overlap] + (1 - alpha) * segments[i-1][:, :, -overlap:]
            else:
                alpha = alpha.view(-1, 1, 1).expand(-1, 512, 6).permute(2,0,1)
                segments[i][:, :overlap, :] = alpha * segments[i][:, :overlap, :] + (1 - alpha) * segments[i-1][:, -overlap:, :]

        # Concatenating all the segments
        if resized_width > 512:
            blended = [segment[:,:,:-overlap] for segment, overlap in zip(segments[:-1], overlaps[1:])] + [segments[-1]]
            merged_image = torch.cat(blended, dim=2)
        else:
            blended = [segment[:,:-overlap,:] for segment, overlap in zip(segments[:-1], overlaps[1:])] + [segments[-1]]
            merged_image = torch.cat(blended, dim=1)

        return merged_image
def remove_flare(image):
    processor=ImageProcessor(model)
    # print(image)
    merge_img = Image.fromarray(image)
    with torch.no_grad():
        output_img=processor.process_image(merge_img).unsqueeze(0)
        gamma=torch.Tensor([2.2])
        deflare_img,_,_=predict_flare_from_6_channel(output_img,gamma)
        deflare_img_np=deflare_img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        deflare_img_pil=Image.fromarray((deflare_img_np * 255).astype(np.uint8))
        flare_img_orig=ImageChops.difference(merge_img.resize(deflare_img_pil.size),deflare_img_pil)
        deflare_img_orig=ImageChops.difference(merge_img,flare_img_orig.resize(merge_img.size,resample=Image.BICUBIC))
            
    torchvision.utils.save_image(deflare_img, './a.jpg')
    ans = Image.open('./a.jpg').convert('RGB')
    return ans
os.system('python setup.py develop')
base_path = './checkpoint'
os.system('apt install git')
os.system('apt install git-lfs')
os.system(f'git clone https://code.openxlab.org.cn/lishenqu/flare_removal_checkpoint.git {base_path}')
# os.system(f'cd {base_path} && git lfs pull')
#load transformers model

# tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
# print(tokenizer)
# # please replace "AutoModelForCausalLM" with your real task
# model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()
# print(model)

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