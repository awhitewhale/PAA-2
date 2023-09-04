from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import os, cv2
import warnings
from niqe import niqe
from PIL import Image
import numpy as np
from natsort import natsorted
warnings.filterwarnings('ignore')


path = 'E:\OneDrive - hust.edu.cn\\files\自然天气图像攻击\图片\图2 正常训练模型'
# methods = natsorted(os.listdir(path))
gtimgs = natsorted(os.listdir(path+'\\'+'RAW'))
for method in ['PAM']:
    # if method == '_gt':
    #     pass
    # else:
    method_path = path + '\\' + method
    pics = natsorted(os.listdir(method_path))
    for i, pic in enumerate(pics):
        if len(pics) == 70:
            desnowed_img = method_path + '\\' + pic
            desnowed_img_path = method_path + '\\' + pic

            gt = path+'\\'+'_gt\\' + gtimgs[int(i/10)]

            desnowed_img = cv2.resize(cv2.imread(desnowed_img), (512, 512))
            gt = cv2.resize(cv2.imread(gt), (512, 512))

            psnr1 = compare_psnr(desnowed_img, gt, data_range=255)
            ssim1 = compare_ssim(desnowed_img, gt, data_range=255, multichannel=True)
            dis = niqe(np.array(Image.open(desnowed_img_path).convert('LA'))[:, :, 0])  # dis
            print('{},{},{},{}'.format(desnowed_img_path.split('\\')[-2:], psnr1, ssim1, dis))
        elif len(pics) == 10 or len(pics) == 7:
            desnowed_img = method_path + '\\' + pic
            desnowed_img_path = method_path + '\\' + pic

            gt = path + '\\' + '_gt\\' + gtimgs[i]

            desnowed_img = cv2.resize(cv2.imread(desnowed_img), (512, 512))
            gt = cv2.resize(cv2.imread(gt), (512, 512))

            psnr1 = compare_psnr(desnowed_img, gt, data_range=255)
            ssim1 = compare_ssim(desnowed_img, gt, data_range=255, multichannel=True)
            dis = niqe(np.array(Image.open(desnowed_img_path).convert('LA'))[:, :, 0])  # dis
            print('{},{},{},{}'.format(desnowed_img_path.split('\\')[-2:], psnr1, ssim1, dis))
        elif len(pics) == 7000:
            desnowed_img = method_path + '\\' + pic
            desnowed_img_path = method_path + '\\' + pic

            gt = path + '\\' + '_gt\\' + gtimgs[int(i/1000)]

            desnowed_img = cv2.resize(cv2.imread(desnowed_img), (512, 512))
            gt = cv2.resize(cv2.imread(gt), (512, 512))

            psnr1 = compare_psnr(desnowed_img, gt, data_range=255)
            ssim1 = compare_ssim(desnowed_img, gt, data_range=255, multichannel=True)
            dis = niqe(np.array(Image.open(desnowed_img_path).convert('LA'))[:, :, 0])  # dis
            print('{},{},{},{}'.format(desnowed_img_path.split('\\')[-2:], psnr1, ssim1, dis))

