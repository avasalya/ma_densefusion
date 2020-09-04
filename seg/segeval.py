from segnet import SegNet as segnet
import torch 
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import glob
import sys
# sys.path.insert(0,'/home/fapsros/anaconda3/lib/python3.7/site-packages')
import cv2



path = '/home/aist/ma_densefusion/'

model = segnet()
model.cuda()
# model.load_state_dict(torch.load(path + 'trained_models/seg_model_96_0.0005561994192357816.pth'))
model.load_state_dict(torch.load(path + 'trained_models/02/seg_model_498_0.00011372075986017194.pth'))

# print(model)
model.eval()

colors = [np.array(Image.open(file).convert("RGB")) for file in sorted(glob.glob(path + 'seg/segmentation/seg_result/testrgb/*.png')) ]
colors_trans = [np.transpose(rgb, (2, 0, 1)) for rgb in colors]
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
colors_norm = [ norm(torch.from_numpy(rgb.astype(np.float32))) for rgb in colors_trans]

# rgb = np.array(Image.open(path + 'seg/segmentation/rgb/3000.png'))
# rgb = np.transpose(rgb, (2, 0, 1))
# norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
#print(rgb)



for idx, rgb in enumerate(colors_norm):
    # rgb = np.transpose(rgb, (2, 0, 1))
    # rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
    rgb = Variable(rgb).cuda()
    semantic = model(rgb.unsqueeze(0))
    _, pred = torch.max(semantic, dim=1)
    pred = pred*255
    # pred = np.transpose(pred, (1, 2, 0))  # (CxHxW)->(HxWxC)
    # ret, threshold = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    torchvision.utils.save_image(pred, path + 'seg/segmentation/seg_result/' + str(idx) + '.png')


# img = np.transpose(pred.unsqueeze(0).cpu().numpy(), (1, 2, 0))
# ret, threshold = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)    
# torchvision.utils.save_image(pred, path + 'seg/segmentation/seg_result/' + str(idx) + '.png')
#print(semantic)
#print(semantic.shape)
# print(pred)
