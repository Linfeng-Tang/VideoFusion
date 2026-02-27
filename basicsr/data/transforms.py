import cv2
import random
import torch
import numpy as np


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.
    It crops lists of lq and gt images with corresponding locations.
    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.
    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def paired_random_crop_multi(img_ir_gts, img_ir_lqs, img_vi_gts, img_vi_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop for both IR and VI images. Support Numpy array and Tensor inputs.
    It crops lists of LQ and GT images with corresponding locations for both IR and VI.
    
    Args:
        img_ir_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT IR images.
        img_ir_lqs (list[ndarray] | ndarray): LQ IR images.
        img_vi_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT VI images.
        img_vi_lqs (list[ndarray] | ndarray): LQ VI images.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.
        
    Returns:
        tuple: (cropped GT IR images, cropped LQ IR images, cropped GT VI images, cropped LQ VI images)
    """

    # Convert inputs to lists if they are not already
    if not isinstance(img_ir_gts, list):
        img_ir_gts = [img_ir_gts]
    if not isinstance(img_ir_lqs, list):
        img_ir_lqs = [img_ir_lqs]
    if not isinstance(img_vi_gts, list):
        img_vi_gts = [img_vi_gts]
    if not isinstance(img_vi_lqs, list):
        img_vi_lqs = [img_vi_lqs]

    # Determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_ir_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq_ir, w_lq_ir = img_ir_lqs[0].size()[-2:]
        h_gt_ir, w_gt_ir = img_ir_gts[0].size()[-2:]
        h_lq_vi, w_lq_vi = img_vi_lqs[0].size()[-2:]
        h_gt_vi, w_gt_vi = img_vi_gts[0].size()[-2:]
    else:
        h_lq_ir, w_lq_ir = img_ir_lqs[0].shape[0:2]
        h_gt_ir, w_gt_ir = img_ir_gts[0].shape[0:2]
        h_lq_vi, w_lq_vi = img_vi_lqs[0].shape[0:2]
        h_gt_vi, w_gt_vi = img_vi_gts[0].shape[0:2]

    lq_patch_size = gt_patch_size // scale

    # Check for scale mismatches
    if h_gt_ir != h_lq_ir * scale or w_gt_ir != w_lq_ir * scale:
        raise ValueError(f'Scale mismatches for IR. GT ({h_gt_ir}, {w_gt_ir}) is not {scale}x '
                         f'multiplication of LQ ({h_lq_ir}, {w_lq_ir}).')
    if h_gt_vi != h_lq_vi * scale or w_gt_vi != w_lq_vi * scale:
        raise ValueError(f'Scale mismatches for VI. GT ({h_gt_vi}, {w_gt_vi}) is not {scale}x '
                         f'multiplication of LQ ({h_lq_vi}, {w_lq_vi}).')
                         
    if h_lq_ir < lq_patch_size or w_lq_ir < lq_patch_size:
        raise ValueError(f'LQ IR ({h_lq_ir}, {w_lq_ir}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')
                         
    if h_lq_vi < lq_patch_size or w_lq_vi < lq_patch_size:
        raise ValueError(f'LQ VI ({h_lq_vi}, {w_lq_vi}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # Randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq_ir - lq_patch_size)
    left = random.randint(0, w_lq_ir - lq_patch_size)

    # Crop LQ patches
    if input_type == 'Tensor':
        img_ir_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_ir_lqs]
        img_vi_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_vi_lqs]
    else:
        img_ir_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_ir_lqs]
        img_vi_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_vi_lqs]

    # Crop corresponding GT patches
    top_gt_ir, left_gt_ir = int(top * scale), int(left * scale)
    top_gt_vi, left_gt_vi = int(top * scale), int(left * scale)

    if input_type == 'Tensor':
        img_ir_gts = [v[:, :, top_gt_ir:top_gt_ir + gt_patch_size, left_gt_ir:left_gt_ir + gt_patch_size] for v in img_ir_gts]
        img_vi_gts = [v[:, :, top_gt_vi:top_gt_vi + gt_patch_size, left_gt_vi:left_gt_vi + gt_patch_size] for v in img_vi_gts]
    else:
        img_ir_gts = [v[top_gt_ir:top_gt_ir + gt_patch_size, left_gt_ir:left_gt_ir + gt_patch_size, ...] for v in img_ir_gts]
        img_vi_gts = [v[top_gt_vi:top_gt_vi + gt_patch_size, left_gt_vi:left_gt_vi + gt_patch_size, ...] for v in img_vi_gts]

    # Return results
    if len(img_ir_gts) == 1:
        img_ir_gts = img_ir_gts[0]
    if len(img_ir_lqs) == 1:
        img_ir_lqs = img_ir_lqs[0]
    if len(img_vi_gts) == 1:
        img_vi_gts = img_vi_gts[0]
    if len(img_vi_lqs) == 1:
        img_vi_lqs = img_vi_lqs[0]
        
    return img_ir_gts, img_ir_lqs, img_vi_gts, img_vi_lqs

def paired_random_crop_vsm(img_ir_gts, img_ir_lqs, img_vi_gts, img_vi_lqs, img_ir_vsms, img_vi_vsms, gt_patch_size, scale, gt_path=None):
    """Paired random crop for both IR and VI images. Support Numpy array and Tensor inputs.
    It crops lists of LQ and GT images with corresponding locations for both IR and VI.
    
    Args:
        img_ir_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT IR images.
        img_ir_lqs (list[ndarray] | ndarray): LQ IR images.
        img_vi_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT VI images.
        img_vi_lqs (list[ndarray] | ndarray): LQ VI images.
        img_ir_vsms (list[ndarray] | ndarray): VSM IR images.
        img_vi_vsms (list[ndarray] | ndarray): VSM VI images.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.
        
    Returns:
        tuple: (cropped GT IR images, cropped LQ IR images, cropped VSM IR images, 
               cropped GT VI images, cropped LQ VI images, cropped VSM VI images)
    """

    # Convert inputs to lists if they are not already
    if not isinstance(img_ir_gts, list):
        img_ir_gts = [img_ir_gts]
    if not isinstance(img_ir_lqs, list):
        img_ir_lqs = [img_ir_lqs]
    if not isinstance(img_vi_gts, list):
        img_vi_gts = [img_vi_gts]
    if not isinstance(img_vi_lqs, list):
        img_vi_lqs = [img_vi_lqs]
    if not isinstance(img_ir_vsms, list):
        img_ir_vsms = [img_ir_vsms]
    if not isinstance(img_vi_vsms, list):
        img_vi_vsms = [img_vi_vsms]

    # Determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_ir_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq_ir, w_lq_ir = img_ir_lqs[0].size()[-2:]
        h_gt_ir, w_gt_ir = img_ir_gts[0].size()[-2:]
        h_lq_vi, w_lq_vi = img_vi_lqs[0].size()[-2:]
        h_gt_vi, w_gt_vi = img_vi_gts[0].size()[-2:]
    else:
        h_lq_ir, w_lq_ir = img_ir_lqs[0].shape[0:2]
        h_gt_ir, w_gt_ir = img_ir_gts[0].shape[0:2]
        h_lq_vi, w_lq_vi = img_vi_lqs[0].shape[0:2]
        h_gt_vi, w_gt_vi = img_vi_gts[0].shape[0:2]

    lq_patch_size = gt_patch_size // scale

    # Check for scale mismatches
    if h_gt_ir != h_lq_ir * scale or w_gt_ir != w_lq_ir * scale:
        raise ValueError(f'Scale mismatches for IR. GT ({h_gt_ir}, {w_gt_ir}) is not {scale}x '
                         f'multiplication of LQ ({h_lq_ir}, {w_lq_ir}).')
    if h_gt_vi != h_lq_vi * scale or w_gt_vi != w_lq_vi * scale:
        raise ValueError(f'Scale mismatches for VI. GT ({h_gt_vi}, {w_gt_vi}) is not {scale}x '
                         f'multiplication of LQ ({h_lq_vi}, {w_lq_vi}).')
                         
    if h_lq_ir < lq_patch_size or w_lq_ir < lq_patch_size:
        raise ValueError(f'LQ IR ({h_lq_ir}, {w_lq_ir}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')
                         
    if h_lq_vi < lq_patch_size or w_lq_vi < lq_patch_size:
        raise ValueError(f'LQ VI ({h_lq_vi}, {w_lq_vi}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # Randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq_ir - lq_patch_size)
    left = random.randint(0, w_lq_ir - lq_patch_size)

    # Crop LQ patches
    if input_type == 'Tensor':
        img_ir_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_ir_lqs]
        img_vi_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_vi_lqs]
        img_ir_vsms = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_ir_vsms]
        img_vi_vsms = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_vi_vsms]
    else:
        img_ir_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_ir_lqs]
        img_vi_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_vi_lqs]
        img_ir_vsms = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_ir_vsms]
        img_vi_vsms = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_vi_vsms]

    # Crop corresponding GT patches
    top_gt_ir, left_gt_ir = int(top * scale), int(left * scale)
    top_gt_vi, left_gt_vi = int(top * scale), int(left * scale)

    if input_type == 'Tensor':
        img_ir_gts = [v[:, :, top_gt_ir:top_gt_ir + gt_patch_size, left_gt_ir:left_gt_ir + gt_patch_size] for v in img_ir_gts]
        img_vi_gts = [v[:, :, top_gt_vi:top_gt_vi + gt_patch_size, left_gt_vi:left_gt_vi + gt_patch_size] for v in img_vi_gts]
    else:
        img_ir_gts = [v[top_gt_ir:top_gt_ir + gt_patch_size, left_gt_ir:left_gt_ir + gt_patch_size, ...] for v in img_ir_gts]
        img_vi_gts = [v[top_gt_vi:top_gt_vi + gt_patch_size, left_gt_vi:left_gt_vi + gt_patch_size, ...] for v in img_vi_gts]

    # Return results
    if len(img_ir_gts) == 1:
        img_ir_gts = img_ir_gts[0]
    if len(img_ir_lqs) == 1:
        img_ir_lqs = img_ir_lqs[0]
    if len(img_vi_gts) == 1:
        img_vi_gts = img_vi_gts[0]
    if len(img_vi_lqs) == 1:
        img_vi_lqs = img_vi_lqs[0]
    if len(img_ir_vsms) == 1:
        img_ir_vsms = img_ir_vsms[0]
    if len(img_vi_vsms) == 1:
        img_vi_vsms = img_vi_vsms[0]
        
    return img_ir_gts, img_ir_lqs, img_vi_gts, img_vi_lqs, img_ir_vsms, img_vi_vsms

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img
