import argparse
import torch
import torchvision
from torch import nn
from torchvision.transforms import functional as F
import os
import numpy as np
import torch
from transform_main import TransSAR, TransSARV2, TransSARV3
import cv2



parser = argparse.ArgumentParser(description='TransSAR')

parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')

parser.add_argument('--load', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--save_path', default='./results/', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--network', type=str,default='TransSARV2',
                    help='network name')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--loadmodel', default='', type=str)



args = parser.parse_args()


modelname = args.model
loaddirec = args.loadmodel
save_path = args.save_path




device = torch.device("cuda")



model = TransSARV2()


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)


model.load_state_dict(torch.load(loaddirec))
model.eval()



if not os.path.isdir(save_path):
                
    os.makedirs(save_path)


im_file = 'test_image.png'

img = cv2.imread(im_file,0) 
noisy_im = (np.float32(img)+1.0)/256.0



x = np.float32(noisy_im)
x = F.to_tensor(x)
x = x.to(device)  # 将输入数据 x 移动到 CUDA 设备上，以匹配模型的设备
x = x.unsqueeze(0)

# print(x)

pred_im = model(x)

# print(pred_im)

tmp = pred_im.detach().cpu().numpy()

tmp = tmp.squeeze()


filename_out = 'test.png'
filepath_out = save_path + filename_out

cv2.imwrite(filepath_out,tmp)



