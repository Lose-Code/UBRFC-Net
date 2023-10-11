import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import transforms

from DCT import BlurDetector
from GAN import Generator
from Parameter import Parameter

device = "cuda:5"
generator = Generator().to(device)
parameter = Parameter().to(device)
generator.load_state_dict(torch.load("/home/wy/code/datasets/cloudy/model_1540")["generator_net"])
parameter.load_state_dict(torch.load("/data/wy/code/datasets/cloudy/model_1540")["parameterNet_net"])

root = '/data/wy/code/datasets/thin_cloudy/test/hazy'
save_path = '/home/wy/code/WY/images/test_img'

def get_transforms():
    all_transforms = [transforms.ToTensor()]

    return transforms.Compose(all_transforms)


with torch.no_grad():
    generator.eval()
    parameter.eval()

    for x in os.listdir(root):

        print(x)
        x_input = Image.open(os.path.join(root, x))
        x_input_tensor = get_transforms()(x_input)
        x_input_tensor = torch.unsqueeze(x_input_tensor, 0)
        x_input_tensor = x_input_tensor.to(device)
        start = time.time()
        pred1 = generator(x_input_tensor)
        (haze,atp,zz), pred2 = parameter(x_input_tensor, pred1)
        bd = BlurDetector()
        image_pred1 = pred1.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        image_pred2 = pred2.squeeze().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        result1, image1 = bd.check_image_size(image_pred1)
        result2, image2 = bd.check_image_size(image_pred2)
        blur1 = bd.get_blurness(image1)
        blur2 = bd.get_blurness(image2)
        print('end:', time.time() - start)
        #print("Blurness1: {:.5f}".format(blur1), "Blurness2: {:.5f}".format(blur2))

        if blur1 >= blur2:
            final_result = torch.cat([x_input_tensor.data, pred2.data], dim=0)
            save_image(pred2.data, os.path.join(save_path,x))
            save_image(final_result.data, os.path.join(save_path, "fianl_"+x))
        else:
            final_result = torch.cat([x_input_tensor.data, pred1.data], dim=0)
            save_image(final_result.data, os.path.join(save_path, "fianl_"+x))
            save_image(pred1.data, os.path.join(save_path,x))

