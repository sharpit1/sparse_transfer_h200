import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from generators_modify import GeneratorResnet
import random

from tqdm import tqdm

import logging
logger = logging.getLogger('logger')

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser(description='Training EGS_TSSA for generating sparse adversarial examples')
parser.add_argument('--train_dir', default='/home/dataset/imagenet-1k/train', help='path to imagenet training set')
parser.add_argument('--model_type', type=str, default='res50', help='Model against GAN is trained: incv3, res50')
parser.add_argument(
    '--generator_mode',
    '--generator-mode',
    dest='generator_mode',
    choices=['legacy', 'isolated'],
    default='legacy',
    help=(
        'legacy uses the learned GPG encoders; isolated uses a private frozen '
        'copy of the ImageNet ResNet-50 layer1 prefix'
    ),
)
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
parser.add_argument('--batch_size', type=int, default=16, help='Number of trainig samples/batch')
parser.add_argument('--sample_per_class', '--samples_per_class', dest='sample_per_class', type=int, default=0,
                    help='maximum number of training samples per class; 0 uses all samples')
parser.add_argument('--n_iters', type=int, default=5, help='Number of PGD attack iteration')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=2.25e-5, help='Initial learning rate for adam')

parser.add_argument('--lam_1', type=float, default=0.0001, help='spa')
parser.add_argument('--lam_2', type=float, default=0.0001, help='spa')
parser.add_argument('--lam_3', type=float, default=0.0001, help='spa')

parser.add_argument('--pb', default='full', type=str, choices=['full', 'half'])
parser.add_argument('--load_CP', default='New', type=str, choices=['New', 'Continue'])
parser.add_argument('--CP_path', type=str, default='', help='path to checkpoint')
parser.add_argument('--out-dir', default='Train_GPG', type=str, help='Output directory')

args = parser.parse_args()
if args.generator_mode == 'isolated' and args.model_type != 'res50':
    parser.error('--generator_mode isolated requires --model_type res50')
torch.set_num_threads(3)

generator_mode_suffix = (
    '' if args.generator_mode == 'legacy' else '_gen_isolated'
)
output_path = os.path.join(args.out_dir, f'GPG_{args.model_type}_tar_{args.target}_eps_{args.eps}_Load_{args.load_CP}_lam1_{args.lam_1}_lam3_{args.lam_3}_pb_{args.pb}{generator_mode_suffix}')
if not os.path.exists(output_path):
    os.makedirs(output_path)
logfile = os.path.join(output_path, f'train_info.log')
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(output_path, f'train_info.log'))
logger.info(args)

eps = args.eps
epochs = args.epochs

# Input dimensions
if args.model_type in ['res50']:
    scale_size = 256
    img_size = 224
    filterSize = 8
    stride = 8
else:
    scale_size = 300
    img_size = 299
    filterSize = 13
    stride = 13

# Model
if args.model_type == 'incv3':
    model = torchvision.models.inception_v3(pretrained=True)
elif args.model_type == 'res50':
    model = torchvision.models.resnet50(pretrained=True)

# Generator
netG = GeneratorResnet(
    inception=args.model_type == 'incv3',
    encoder_mode=args.generator_mode,
    encoder_backbone=model if args.generator_mode == 'isolated' else None,
)
if args.load_CP == 'Continue':
    checkpoint = torch.load(args.CP_path, map_location='cpu')
    # Preserve the historical relaxed load only for legacy high-budget runs.
    # Isolated mode must fail closed on an incompatible encoder checkpoint.
    strict_checkpoint = not (
        args.generator_mode == 'legacy' and args.eps > 128
    )
    netG.load_state_dict(checkpoint, strict=strict_checkpoint)
model = model.cuda()
model.eval()
netG.cuda()

# Optimizer
optimG = optim.Adam(
    list(netG.trainable_parameters()),
    lr=args.lr,
    betas=(0.5, 0.999),
)

# Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size, antialias=True),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


train_set = datasets.ImageFolder(args.train_dir, data_transform)
if args.sample_per_class > 0:
    class_to_indices = defaultdict(list)
    for idx, (_, class_idx) in enumerate(train_set.samples):
        class_to_indices[class_idx].append(idx)

    rng = random.Random(seed)
    selected_indices = []
    for class_idx in sorted(class_to_indices):
        indices = class_to_indices[class_idx]
        rng.shuffle(indices)
        selected_indices.extend(indices[:args.sample_per_class])
    rng.shuffle(selected_indices)

    logger.info('sample_per_class=%d selected %d/%d samples across %d classes',
                args.sample_per_class, len(selected_indices), len(train_set), len(class_to_indices))
    train_set = Subset(train_set, selected_indices)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
train_size = len(train_set)

# Adv Loss
def CWLoss(logits, target, kappa=-0., tar=False):
    target = torch.ones(logits.size(0)).cuda().type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(torch.eye(1000).type(torch.cuda.FloatTensor)[target.long()].cuda())
    real = torch.sum(target_one_hot * logits, 1)
    other = torch.max((1 - target_one_hot) * logits - (target_one_hot * 10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    if tar:
        return torch.sum(torch.max(other - real, kappa))
    else:
        return torch.sum(torch.max(real - other, kappa))


criterion = CWLoss


def attack_pgd(model, x, y, eps, alpha, n_iters):
    delta = torch.zeros_like(x).cuda()
    delta.uniform_(-eps, eps)
    delta = torch.clamp(delta, 0-x, 1-x)

    delta.requires_grad = True
    for _ in range(n_iters):
        output = model(normalize(x+delta))
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        d = torch.clamp(delta + alpha * torch.sign(grad), min=-eps, max=eps)
        d = torch.clamp(d, 0 - x, 1 - x)
        delta.data = d
        delta.grad.zero_()

    return delta.detach()


FR_white_box = []
tra_loss, norm_0, norm_1, norm_2, test = [], [], [], [], []
iterp = 1000 // args.batch_size
i_len = train_size // (iterp * args.batch_size)

for epoch in range(epochs):
    FR_wb, FR_wb_epoch = 0, 0

    if args.load_CP == 'New':
        if epoch < 3:
            lam_1 = 0.00
            lam_2 = args.lam_2
            lam_3 = args.lam_3
        else:
            lam_1 = args.lam_1
            lam_2 = args.lam_2
            lam_3 = args.lam_3
    elif args.load_CP == 'Continue':
        lam_1 = args.lam_1
        lam_2 = args.lam_2
        lam_3 = args.lam_3
    else:
        raise ValueError

    for i, (img, gt) in enumerate(tqdm(train_loader)):
        img = img.cuda()
        gt = gt.cuda()

        if args.target == -1:
            img_in = normalize(img.clone().detach())
            out = model(img_in)
            label = out.argmax(dim=-1).detach()
            out_wb = label.clone().detach()
            out.backward(torch.ones_like(out))
        else:
            out = torch.LongTensor(img.size(0))
            out.fill_(args.target)
            label = out.cuda()

            out_tmp = model(normalize(img.clone().detach()))
            out_tmp.backward(torch.ones_like(out_tmp))
            out_wb = label.clone().detach()

        netG.train()
        optimG.zero_grad()

        if args.pb == 'half':
            if args.eps > 128:
                grad_eps = (args.eps / 2.)
                grad_alpha = (grad_eps / args.n_iters)
            else:
                grad_eps = args.eps
                grad_alpha = (grad_eps / args.n_iters)
        elif args.pb == 'full':
            grad_eps = args.eps
            grad_alpha = (grad_eps / args.n_iters)
        else:
            raise ValueError

        grad_delta = attack_pgd(model, img, gt, eps=grad_eps/255., alpha=grad_alpha/255., n_iters=args.n_iters)
        x_grad_adv = img + grad_delta
        if args.generator_mode == 'legacy':
            adv, adv_inf, adv_0, adv_00, diff_spa_grad = netG(
                img, args.eps/255., x_grad_adv
            )
        else:
            # A frozen clean encoder makes the legacy feature-alignment term
            # constant with respect to the decoder.  Keep pixel-space PGD
            # guidance and disable only that ineffective term.
            adv, adv_inf, adv_0, adv_00 = netG(img, args.eps/255.)
            diff_spa_grad = adv.new_zeros(())

        # Gradient Regularization
        grad_guided_loss = torch.sum((adv_inf - grad_delta) ** 2)

        adv_img = adv.clone().detach()
        adv_out = model(normalize(adv))
        adv_out_to_wb = adv_out.clone().detach()

        if args.target == -1:
            FR_wb_tmp = torch.sum(adv_out_to_wb.argmax(dim=-1) != out_wb).item()
            # Untargeted Attack
            loss_adv = criterion(adv_out, label)
        else:
            FR_wb_tmp = torch.sum(adv_out_to_wb.argmax(dim=-1) == out_wb).item()
            # Targeted Attack
            loss_adv = criterion(adv_out, label, tar=True)

        FR_wb += FR_wb_tmp
        FR_wb_epoch += FR_wb_tmp

        loss_spa = torch.norm(adv_0, 1)
        bi_adv_00 = torch.where(adv_00 < 0.5, torch.zeros_like(adv_00), torch.ones_like(adv_00))
        loss_qua = torch.sum((bi_adv_00 - adv_00) ** 2)
        loss = loss_adv + lam_1 * loss_spa + lam_2 * loss_qua + lam_3 * grad_guided_loss + lam_3 * diff_spa_grad

        loss.backward()
        optimG.step()

        adv_loss = loss_adv
        spa1 = lam_1 * loss_spa
        spa2 = lam_2 * loss_qua

        if i % iterp == 0:
            FR = FR_wb / (iterp * args.batch_size)
            FR_wb = 0
            adv_0_img = torch.where(adv_0 < 0.5, torch.zeros_like(adv_0), torch.ones_like(adv_0)).clone().detach()
            l0 = (torch.norm(adv_0_img.clone().detach(), 0) / args.batch_size).item()
            l1 = (torch.norm(adv_0_img.clone().detach() * adv_inf.clone().detach(), 1) / args.batch_size).item()
            l2 = (torch.norm(adv_0_img.clone().detach() * adv_inf.clone().detach(), 2) / args.batch_size).item()
            linf = (torch.norm(adv_0_img.clone().detach() * adv_inf.clone().detach(), p=np.inf)).item()
            tra_loss.append(loss.item())
            FR_white_box.append(FR)
            norm_0.append(l0)
            norm_1.append(l1)
            norm_2.append(l2)

        if i % 2000 == 0:
            Progress_info = str(i) + ' - ' + str(train_size // args.batch_size)
            logger.info('Epoch \t Progress \t l0 \t l1  \t l2 \t linf')
            logger.info('%d \t %s \t %.2f\t %.2f \t %.2f \t %.2f', epoch, Progress_info, l0, l1, l2, linf)
            logger.info('loss \t adv \t spa1 \t spa2 \t FR')
            logger.info('%.3f \t %.3f \t %.3f\t %.3f \t %.3f', loss.item(), adv_loss.item(), spa1.item(), spa2.item(), FR)
            logger.info('')


        if i in [100, 10000, 20000, 30000]:
            vutils.save_image(vutils.make_grid(adv_img, normalize=True, scale_each=True), os.path.join(output_path, 'ep{}_adv{}.png'.format(epoch, i)))
            vutils.save_image(vutils.make_grid(adv_img - img, normalize=True, scale_each=True), os.path.join(output_path, 'ep{}_noise{}.png'.format(epoch, i)))
            # vutils.save_image(vutils.make_grid(img, normalize=True, scale_each=True), os.path.join(output_path, 'ep{}_org{}.png'.format(epoch, i)))

    FR_wb_ep_mean = FR_wb_epoch / train_size
    print('running:{} | FR-{}:{}\n'.format(epoch, args.model_type, FR_wb_ep_mean))
    start, end = int(epoch) * i_len, int(epoch + 1) * i_len
    N0 = np.mean(norm_0[start:end])
    N1 = np.mean(norm_1[start:end])
    try:
        print('loss:{}--L0:{}--L1:{}--L2:{}\n'.format(tra_loss[-1], N0, N1, np.mean(norm_2[start:end])))
    except:
        pass

    try:
        save_path = 'GN_{}_{}.pth'.format(args.model_type, epoch)
        torch.save(netG.state_dict(), os.path.join(output_path, save_path))
    except:
        save_path = output_path + '/' + 'GN_{}_{}.pth'.format(args.model_type, epoch)
        torch.save(netG.state_dict(), os.path.join(save_path))
