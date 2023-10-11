import os.path
import numpy as np
from tqdm import tqdm
from Option import opt
from Declaration import device,generator,parameterNet,test_dataloader
from Metrics import ssim, psnr
from torchvision.utils import save_image
import math
import torch
def padding_image(image, h, w):
    assert h >= image.size(2)
    assert w >= image.size(3)
    padding_top = (h - image.size(2)) // 2
    padding_down = h - image.size(2) - padding_top
    padding_left = (w - image.size(3)) // 2
    padding_right = w - image.size(3) - padding_left
    out = torch.nn.functional.pad(image, (padding_left, padding_right, padding_top, padding_down), mode='reflect')
    return out, padding_left, padding_left + image.size(3), padding_top, padding_top + image.size(2)

def test(genertor_net, parameter_net, loader_test):
    genertor_net.eval()
    parameter_net.eval()
    clear_psnrs,clear_ssims = [],[]
    for i, (inputs, targets,name) in tqdm(enumerate(loader_test), total=len(loader_test), leave=False,desc="测试中"):

        h, w = inputs.shape[2], inputs.shape[3]
        
        
        if h>w:
            max_h = int(math.ceil(h / 512)) * 512
            max_w = int(math.ceil(w / 512)) * 512
        else:
            max_h = int(math.ceil(h / 256)) * 256
            max_w = int(math.ceil(w / 256)) * 256
        inputs, ori_left, ori_right, ori_top, ori_down = padding_image(inputs, max_h, max_w)

        inputs = inputs.to(device)
        targets = targets.to(device)

        pred = genertor_net(inputs)
        _, dehazy_pred = parameter_net(inputs, pred)

        
        pred = pred.data[:, :, ori_top:ori_down, ori_left:ori_right]
        dehazy_pred = dehazy_pred.data[:, :, ori_top:ori_down, ori_left:ori_right]

        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)

        ssim11 = ssim(dehazy_pred, targets).item()
        psnr11 = psnr(dehazy_pred, targets)

        save_image(inputs.data[:1], os.path.join(opt.out_hazy_path,"%s.png" % name[0]))
        save_image(targets.data[:1], os.path.join(opt.out_gt_path,"%s.png" % name[0]))

        if psnr11 >= psnr1:
            save_image(dehazy_pred.data[:1], os.path.join(opt.out_clear_path,"%s.png" % name[0]))
            clear_ssims.append(ssim11)
            clear_psnrs.append(psnr11)
        else:
            save_image(pred.data[:1], os.path.join(opt.out_clear_path,"%s.png" % name[0]))
            clear_ssims.append(ssim1)
            clear_psnrs.append(psnr1)

    return np.mean(clear_ssims), np.mean(clear_psnrs)

if __name__ == '__main__':
    print(test(generator,parameterNet,test_dataloader))