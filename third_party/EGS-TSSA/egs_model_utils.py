import sys
import types
from pathlib import Path

import torch
import torchvision


def canonical_model_type(name):
    key = name.lower().replace("-", "_")
    aliases = {
        "inception_v3": "incv3",
        "incv3": "incv3",
        "resnet50": "res50",
        "res50": "res50",
        "vit": "vit_b16",
        "vit_b16": "vit_b16",
        "vim_small": "vim_small",
    }
    if key not in aliases:
        raise ValueError("Unsupported EGS source model_type: {}".format(name))
    return aliases[key]


def is_inception_model(name):
    return canonical_model_type(name) == "incv3"


def load_vit_b16():
    try:
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        return torchvision.models.vit_b_16(weights=weights)
    except AttributeError:
        return torchvision.models.vit_b_16(pretrained=True)


def _install_timm_compat_shims():
    try:
        import timm.models._builder  # noqa: F401
    except ModuleNotFoundError:
        from timm.models import helpers as timm_helpers

        builder_module = types.ModuleType("timm.models._builder")

        def resolve_pretrained_cfg(pretrained_cfg=None, kwargs=None, default_cfg=None, **_):
            if pretrained_cfg is not None:
                return pretrained_cfg
            if isinstance(kwargs, dict) and kwargs.get("pretrained_cfg") is not None:
                return kwargs["pretrained_cfg"]
            if default_cfg is not None:
                return default_cfg
            return {}

        builder_module.resolve_pretrained_cfg = resolve_pretrained_cfg
        builder_module._update_default_kwargs = timm_helpers.update_default_cfg_and_kwargs
        builder_module._update_default_model_kwargs = timm_helpers.update_default_cfg_and_kwargs
        sys.modules["timm.models._builder"] = builder_module


def _resolve_vim_checkpoint(vim_model_root, filename):
    for directory in (vim_model_root / "ckpts", vim_model_root / "ckpt"):
        path = directory / filename
        if path.exists():
            return path
    return vim_model_root / "ckpts" / filename


def load_vim_small():
    from timm import create_model

    _install_timm_compat_shims()
    third_party_root = Path(__file__).resolve().parent.parent
    vim_root = third_party_root / "Vim"
    vim_model_root = vim_root / "vim"
    mamba_root = vim_root / "mamba-1p1p1"

    for path in (mamba_root, vim_model_root):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)

    import models_mamba  # noqa: F401

    model = create_model(
        model_name="vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2",
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=224,
    )
    ckpt_path = _resolve_vim_checkpoint(vim_model_root, "vim_s_midclstok_80p5acc.pth")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    return model


def load_classifier_model(name):
    key = name.lower().replace("-", "_")
    if key in {"incv3", "inception_v3"}:
        return torchvision.models.inception_v3(pretrained=True)
    if key in {"res50", "resnet50"}:
        return torchvision.models.resnet50(pretrained=True)
    if key == "dense161":
        return torchvision.models.densenet161(pretrained=True)
    if key == "vgg16":
        return torchvision.models.vgg16(pretrained=True)
    if key in {"vit", "vit_b16"}:
        return load_vit_b16()
    if key == "vim_small":
        return load_vim_small()
    raise ValueError("Unsupported classifier model: {}".format(name))


def _cam_layer_for_source(model, model_type):
    if model_type == "incv3":
        return model.Mixed_7c
    if model_type == "res50":
        return model.layer4[-1]
    if model_type == "vit_b16":
        return model.encoder.layers[-2]
    if model_type == "vim_small":
        return model.layers[-2]
    raise ValueError("Unsupported EGS source model_type: {}".format(model_type))


def load_source_model(model_type, forward_hook, backward_hook):
    model_type = canonical_model_type(model_type)
    model = load_classifier_model(model_type)
    cam_layer = _cam_layer_for_source(model, model_type)
    cam_layer.register_forward_hook(forward_hook)
    cam_layer.register_full_backward_hook(backward_hook)
    return model, model_type


def _sequence_to_grid(tensor, cls_token_position):
    if tensor.shape[-1] >= tensor.shape[1]:
        tensor = tensor.transpose(1, 2).contiguous()

    tokens = tensor.shape[-1]
    side = int(tokens ** 0.5)
    if side * side == tokens:
        return tensor.reshape(tensor.shape[0], tensor.shape[1], side, side)

    trim_side = int((tokens - 1) ** 0.5)
    if tokens > 1 and trim_side * trim_side == tokens - 1:
        if cls_token_position == "front":
            tensor = tensor[:, :, 1:]
        elif cls_token_position == "middle":
            mid = tokens // 2
            tensor = torch.cat((tensor[:, :, :mid], tensor[:, :, mid + 1:]), dim=-1)
        elif cls_token_position == "back":
            tensor = tensor[:, :, :-1]
        else:
            raise ValueError("Unknown class token position: {}".format(cls_token_position))
        return tensor.reshape(tensor.shape[0], tensor.shape[1], trim_side, trim_side)

    raise ValueError("Cannot reshape token feature with {} tokens into a square grid".format(tokens))


def feature_to_spatial_parts(output, model_type):
    if isinstance(output, (tuple, list)):
        parts = []
        for item in output:
            if torch.is_tensor(item):
                parts.extend(feature_to_spatial_parts(item, model_type))
        if not parts:
            raise ValueError("No tensor output found for EGS CAM")
        return parts
    if output.ndim == 4:
        return [output]
    if output.ndim == 3:
        model_type = canonical_model_type(model_type)
        cls_token_position = "middle" if model_type == "vim_small" else "front"
        return [_sequence_to_grid(output, cls_token_position)]
    raise ValueError("Unsupported feature rank for EGS CAM: {}".format(output.ndim))


def feature_to_spatial(output, model_type):
    parts = feature_to_spatial_parts(output, model_type)
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    for item in parts[1:]:
        if item.shape != first.shape:
            raise ValueError("Cannot combine CAM tensors with shapes {} and {}".format(first.shape, item.shape))
        first = first + item
    return first


def cam_from_feature_and_grad(features, grads):
    feature_parts = features if isinstance(features, list) else [features]
    grad_parts = grads if isinstance(grads, list) else [grads]
    if len(feature_parts) != len(grad_parts):
        raise ValueError("Feature/gradient CAM part count mismatch: {} vs {}".format(len(feature_parts), len(grad_parts)))

    cam = None
    for feature, grad in zip(feature_parts, grad_parts):
        if feature.shape != grad.shape:
            raise ValueError("Feature/gradient CAM shape mismatch: {} vs {}".format(feature.shape, grad.shape))
        weights = grad.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        term = (weights * feature).sum(dim=1)
        cam = term if cam is None else cam + term
    return cam
