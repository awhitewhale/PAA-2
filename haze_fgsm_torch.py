import argparse
import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


def opt():
    parser = argparse.ArgumentParser(description='Optimization Parameters')
    parser.add_argument('--batch_size', type=int, default=1, help='How many images process at one time.')
    parser.add_argument('--max_epsilon', type=float, default=16.0, help='max epsilon.')
    parser.add_argument('--num_iter', type=int, default=10, help='max iteration.')
    parser.add_argument('--momentum', type=float, default=1.0, help='momentum about the model.')
    parser.add_argument('--number', type=int, default=20, help='the number of images for variance tuning')
    parser.add_argument('--beta', type=float, default=1.5, help='the bound for variance tuning.')
    parser.add_argument('--image_width', type=int, default=299, help='Width of each input images.')
    parser.add_argument('--image_height', type=int, default=299, help='Height of each input images.')
    parser.add_argument('--prob', type=float, default=0.5, help='probability of using diverse inputs.')
    parser.add_argument('--image_resize', type=int, default=331, help='Height of each input images.')
    parser.add_argument('--checkpoint_path', type=str, default='/home/data/liuyifan/project/ADVATK/Targeted_Patch-wise-plusplus_iterative_attack/models/', help='Path to checkpoint for pretained models.')
    parser.add_argument('--input_dir', type=str, default='/home/data/liuyifan/project/datasets/dehaze/gt2/', help='Input directory with images.')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory with images.')
    FLAGS = parser.parse_args()
    return FLAGS

FLAGS = opt()
def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in glob.glob(os.path.join(input_dir, '*')):
        with open(filepath, 'rb') as f:
            image = np.array(Image.open(f).convert('RGB'), dtype=np.float32) / 255.0
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def input_diversity(input_tensor):
    image_width = FLAGS.image_width
    image_resize = FLAGS.image_resize
    image_height = FLAGS.image_height
    prob = FLAGS.prob

    rnd = torch.randint(image_width, image_resize, (1,)).item()
    rescaled = transforms.Resize((rnd, rnd))(input_tensor)

    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, (1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, (1,)).item()
    pad_right = w_rem - pad_left

    padded = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom))(rescaled)
    padded = transforms.Resize((image_resize, image_resize))(padded)

    ret = padded if torch.rand(1) < prob else input_tensor
    ret = transforms.Resize((image_height, image_width))(ret)

    return ret


def batch_grad(x, one_hot, i, max_iter, alpha, grad):
    x_neighbor = x + torch.empty_like(x).uniform_(-alpha, alpha)
    x_neighbor_2 = 1 / 2. * x_neighbor
    x_neighbor_4 = 1 / 4. * x_neighbor
    x_neighbor_8 = 1 / 8. * x_neighbor
    x_neighbor_16 = 1 / 16. * x_neighbor
    x_res = torch.cat([x_neighbor, x_neighbor_2, x_neighbor_4, x_neighbor_8, x_neighbor_16], dim=0)
    inception_v3 = models.inception_v3(pretrained=True)
    logits_v3, end_points_v3 = inception_v3(input_diversity(x_res), num_classes=1001, is_training=False)
    cross_entropy = F.softmax_cross_entropy(logits_v3, one_hot)
    grad += torch.sum(
        torch.split(torch.autograd.grad(cross_entropy, x_res)[0], 5)
        * torch.tensor([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.], dtype=torch.float32).view(-1, 1, 1, 1),
        dim=0
    )
    i = i + 1
    return x, one_hot, i, max_iter, alpha, grad


def graph(x, y, i, x_max, x_min, grad, variance):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001
    x_nes = x.clone()
    x_batch = torch.cat([x_nes, x_nes / 2., x_nes / 4., x_nes / 8., x_nes / 16.], dim=0)
    logits_v3, end_points_v3 = inception_v3.inception_v3(
        input_diversity(x_batch), num_classes=num_classes, is_training=False)
    pred = torch.argmax(end_points_v3['Predictions'], dim=1)
    first_round = torch.eq(i, 0).to(torch.int64)
    y = first_round * pred[:y.shape[0]] + (1 - first_round) * y
    one_hot = F.one_hot(y, num_classes).float().repeat(5, 1)
    cross_entropy = F.softmax_cross_entropy_with_logits(logits_v3, one_hot)
    new_grad = torch.sum(torch.split(torch.autograd.grad(cross_entropy, x_batch)[0], 5) * torch.tensor([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.])[:, None, None, None, None], dim=0)
    iter = torch.tensor(0)
    max_iter = torch.tensor(FLAGS.number)
    _, _, _, _, _, global_grad = while_loop(grad_finish, batch_grad, [x, one_hot, iter, max_iter, eps * FLAGS.beta, torch.zeros_like(new_grad)])
    current_grad = new_grad + variance
    noise = F.conv2d(current_grad, stack_kernel, stride=1, padding='SAME')
    noise = noise / torch.mean(torch.abs(noise), dim=[1, 2, 3], keepdim=True)
    noise = momentum * grad + noise
    variance = global_grad / (1. * FLAGS.number) - new_grad
    x = x + alpha * torch.sign(noise)
    x = torch.clamp(x, x_min, x_max)
    i = i + 1
    return x, y, i, x_max, x_min, noise, variance
