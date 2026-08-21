import copy

import torch
import torch.nn as nn


ngf = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GENERATOR_ENCODER_MODES = ("legacy", "isolated")


class FrozenResNet50Layer1(nn.Module):
    """Private, frozen ResNet-50 prefix ending at ``layer1``."""

    def __init__(self, resnet50):
        super(FrozenResNet50Layer1, self).__init__()
        required = ("conv1", "bn1", "relu", "maxpool", "layer1")
        missing = [name for name in required if not hasattr(resnet50, name)]
        if missing:
            raise ValueError(
                "isolated generator mode requires a torchvision-style "
                "ResNet-50; missing={}".format(missing)
            )

        # A deep copy prevents classifier training/evaluation state from leaking
        # into the generator encoder or vice versa.
        self.conv1 = copy.deepcopy(resnet50.conv1)
        self.bn1 = copy.deepcopy(resnet50.bn1)
        self.relu = copy.deepcopy(resnet50.relu)
        self.maxpool = copy.deepcopy(resnet50.maxpool)
        self.layer1 = copy.deepcopy(resnet50.layer1)
        self.register_buffer(
            "normalization_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "normalization_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Frozen BatchNorm buffers must remain immutable even when netG.train()
        # is called by the outer training loop.
        super(FrozenResNet50Layer1, self).train(False)
        return self

    def forward(self, image):
        if image.dim() != 4 or image.size(1) != 3:
            raise ValueError("image must have shape Bx3xHxW")
        mean = self.normalization_mean.to(dtype=image.dtype)
        std = self.normalization_std.to(dtype=image.dtype)
        feature = (image - mean) / std
        feature = self.conv1(feature)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        return self.layer1(feature)


class GeneratorResnet(nn.Module):
    """Training generator for GPG.

    This module intentionally keeps the same state-dict names as
    Generator.py so checkpoints trained here can be evaluated by Eval_GPG.py.
    The forward signature matches GPG_train.py:

        netG(clean_image, eps, gradient_adversarial_image)

    During training, the auxiliary gradient encoder implements the paper's
    gradient adversarial feature guidance term:

        L_FG = || E_c(x) - E_a(x + g) ||_2.
    """

    def __init__(
        self,
        inception=False,
        eps=1.0,
        evaluate=False,
        encoder_mode="legacy",
        encoder_backbone=None,
    ):
        super(GeneratorResnet, self).__init__()
        if encoder_mode not in GENERATOR_ENCODER_MODES:
            raise ValueError(
                "encoder_mode must be one of {}".format(GENERATOR_ENCODER_MODES)
            )
        if encoder_mode == "isolated" and encoder_backbone is None:
            raise ValueError(
                "encoder_backbone is required when encoder_mode='isolated'"
            )
        self.inception = inception
        self.encoder_mode = encoder_mode

        if self.encoder_mode == "legacy":
            self.block1 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(
                    ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(
                    ngf * 2,
                    ngf * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            )
        else:
            self.isolated_encoder = FrozenResNet50Layer1(encoder_backbone)

        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)

        if self.encoder_mode == "legacy":
            self.Grad_block1 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
            )
            self.Grad_block2 = nn.Sequential(
                nn.Conv2d(
                    ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
            )
            self.Grad_block3 = nn.Sequential(
                nn.Conv2d(
                    ngf * 2,
                    ngf * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            )

        self.upsampl_inf1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )
        self.upsampl_inf2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )
        self.blockf_inf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
        )

        self.upsampl_01 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        )
        self.upsampl_02 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )
        self.blockf_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 1, kernel_size=7, padding=0),
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)
        self.eps = eps
        self.evaluate = evaluate

    def train(self, mode=True):
        super(GeneratorResnet, self).train(mode)
        if self.encoder_mode == "isolated":
            self.isolated_encoder.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _clean_encode(self, x):
        if self.encoder_mode == "legacy":
            x = self.block1(x)
            x = self.block2(x)
            x_feature = self.block3(x)
        else:
            with torch.no_grad():
                x_feature = self.isolated_encoder(x).detach()
        code = self.resblock1(x_feature)
        code = self.resblock2(code)
        code = self.resblock3(code)
        code = self.resblock4(code)
        code = self.resblock5(code)
        code = self.resblock6(code)
        return x_feature, code

    def _grad_encode(self, x):
        if self.encoder_mode != "legacy":
            raise RuntimeError(
                "gradient feature guidance is disabled in isolated encoder mode"
            )
        x = self.Grad_block1(x)
        x = self.Grad_block2(x)
        return self.Grad_block3(x)

    def forward(self, input, eps=None, grad_AE=None):
        # Support both netG(x, eps, x_adv) and netG(x, x_adv).
        if torch.is_tensor(eps) and grad_AE is None:
            grad_AE = eps
            eps = None

        perturb_budget = self.eps if eps is None else float(eps)

        x_feature, code = self._clean_encode(input)
        loss_fg = None
        if grad_AE is not None:
            code_grad = self._grad_encode(grad_AE)
            loss_fg = torch.norm(x_feature - code_grad, p=2)

        x = self.upsampl_inf1(code)
        x = self.upsampl_inf2(x)
        x = self.blockf_inf(x)
        if self.inception:
            x = self.crop(x)
        x_inf = perturb_budget * torch.tanh(x)

        x = self.upsampl_01(code)
        x = self.upsampl_02(x)
        x = self.blockf_0(x)
        if self.inception:
            x = self.crop(x)
        adv_00 = (torch.tanh(x) + 1) / 2

        if self.evaluate:
            adv_0 = torch.where(
                adv_00 < 0.5,
                torch.zeros_like(adv_00).detach(),
                torch.ones_like(adv_00).detach(),
            )
        else:
            hard_mask = torch.where(
                adv_00 < 0.5,
                torch.zeros_like(adv_00),
                torch.ones_like(adv_00),
            )
            adv_0 = torch.where(torch.rand_like(adv_00) < 0.5, adv_00, hard_mask.detach())

        adv = torch.clamp((x_inf * adv_0) + input, min=0, max=1)

        if grad_AE is None:
            return adv, x_inf, adv_0, adv_00
        return adv, x_inf, adv_0, adv_00, loss_fg


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x):
        return x + self.block(x)
