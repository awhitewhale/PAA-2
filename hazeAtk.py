import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torchvision.models as models
from torch.autograd import Variable as V
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
import argparse
import csv
from torch.utils import data
from torchvision.transforms import ToTensor
from torchvision.utils import save_image



def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default='dataset.csv', help='Input directory with images.')
    parser.add_argument('--input_dir', type=str, default='/home/data/liuyifan/project/datasets/dehaze/gt/', help='Input directory with images.')
    parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
    parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
    parser.add_argument("--max_epsilon", type=float, default=5.0, help="Maximum size of adversarial perturbation.")
    parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
    parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
    parser.add_argument("--image_resize", type=int, default=330, help="Height of each input images.")
    parser.add_argument("--batch_size", type=int, default=100, help="How many images process at one time.")
    parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
    parser.add_argument("--amplification", type=float, default=10.0, help="To amplifythe step size.")
    parser.add_argument("--prob", type=float, default=0.7, help="probability of using diverse inputs.")

    opt = parser.parse_args()
    return opt


opt = args()
torch.backends.cudnn.benchmark = True
transforms = T.Compose([T.CenterCrop(opt.image_width), T.ToTensor()])
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
model.eval()


class ImageNet(data.Dataset):
    def __init__(self, dir, csv_path, transforms = None):
        self.dir = dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId']
        Truelabel = img_obj['TrueLabel']
        TargetClass = img_obj['TargetClass']
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)




class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class Permute(nn.Module):
    def __init__(self, permutation=[2, 1, 0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        return input[:, self.permutation]


def truth_label(img_path):
    img = Image.open(img_path)
    tensor = ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    prob, pred = output.topk(2, dim=1)
    pred = pred.flatten()
    return pred.tolist()

def dataset_initialization(path):
    gt_floder = path + '/gt/'
    trans_floder = path + '/trans/'
    gt_files = [gtimg for gtimg in os.listdir(gt_floder)]
    trans_files = [transimg for transimg in os.listdir(trans_floder)]
    TrueLabel = []
    TargetClass = []
    for index in tqdm(range(len(gt_files))):
        labels = truth_label(gt_floder + gt_files[index])
        TrueLabel.append(labels[0])
        TargetClass.append(labels[1])
    with open('dataset.csv', 'w') as f:
        writer = csv.writer(f)
        header = ['ImageId', 'transImageId', 'TrueLabel', 'TargetClass']
        writer.writerow(header)
        for i in range(len(gt_files)):
            row = [gt_files[i], trans_files[i], TrueLabel[i], TargetClass[i]]
            writer.writerow(row)

def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2

def project_noise(x, stack_kern, padding_size):
    # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = F.conv2d(x, stack_kern, padding = (padding_size, padding_size), groups=3)
    return x

stack_kern, padding_size = project_kern(3)

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def graph(x, gt, x_min, x_max):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    alpha_beta = alpha * opt.amplification
    gamma = alpha_beta

    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                models.inception_v3(pretrained=True).eval().cuda())
    x.requires_grad = True
    amplification = 0.0
    for i in range(num_iter):

        output_v3 = model(x)
        loss = F.cross_entropy(output_v3, gt)
        loss.backward()
        noise = x.grad.data

        # MI-FGSM
        # noise = noise / torch.abs(noise).mean([1,2,3], keepdim=True)
        # noise = momentum * grad + noise
        # grad = noise

        amplification += alpha_beta * torch.sign(noise)
        cut_noise = torch.clamp(abs(amplification) - eps, 0, 10000.0) * torch.sign(amplification)
        projection = gamma * torch.sign(project_noise(cut_noise, stack_kern, padding_size))
        amplification += projection

        # x = x + alpha * torch.sign(noise)
        x = x + alpha_beta * torch.sign(noise) + projection
        x = clip_by_tensor(x, x_min, x_max)
        x = V(x, requires_grad = True)

    return x.detach()


def input_diversity(input_tensor):
    rnd = torch.randint(opt.image_width, opt.image_resize, ())
    rescaled = F.interpolate(input_tensor, size = [rnd, rnd], mode = 'bilinear', align_corners=True)
    h_rem = opt.image_resize - rnd
    w_rem = opt.image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    pad_list = (pad_left, pad_right, pad_top, pad_bottom)
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [opt.image_resize, opt.image_resize])
    return padded if torch.rand(()) < opt.prob else input_tensor


def main():
    res152 = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                 models.resnet152(pretrained=True).eval().cuda())
    inc_v3 = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                 models.inception_v3(pretrained=True).eval().cuda())
    resnext50_32x4d = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                          models.resnext50_32x4d(pretrained=True).eval().cuda())
    dense161 = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                   models.densenet169(pretrained=True).eval().cuda())


    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum_res152, sum_v3, sum_rext, sum_den = 0,0,0,0

    for images, _,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)
        adv_img = graph(images, gt, images_min, images_max)
        save_image(adv_img, 'output2/{}'.format(_[0]))

        with torch.no_grad():
            sum_res152 += (res152(adv_img).argmax(1) != gt).detach().sum().cpu()
            sum_v3 += (inc_v3(adv_img).argmax(1) != gt).detach().sum().cpu()
            sum_rext += (resnext50_32x4d(adv_img).argmax(1) != gt).detach().sum().cpu()
            sum_den += (dense161(adv_img).argmax(1) != gt).detach().sum().cpu()

    print('inc_v3 = {:.2%}'.format(sum_v3 / 1000.0))
    print('res152 = {:.2%}'.format(sum_res152 / 1000.0))
    print('dense = {:.2%}'.format(sum_den / 1000.0))
    print('rext = {:.2%}'.format(sum_rext / 1000.0))


if __name__ == '__main__':
    dataset_initialization('/home/data/liuyifan/project/datasets/dehaze/')

    # main()
