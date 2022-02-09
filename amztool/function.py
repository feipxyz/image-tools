from typing import Any, Callable, List, Optional, Tuple, TypeVar
import cv2
import numpy as np
import random
from math import fabs
from math import sin
from math import radians
from math import cos

import warnings


def get_rotate_lossless_matrix(angle, width, height):
    height_new = int(width * fabs(sin(radians(angle))) + height*fabs(cos(radians(angle))))
    width_new = int(height * fabs(sin(radians(angle))) + width*fabs(cos(radians(angle))))
    mat_rotation = cv2.getRotationMatrix2D((width/2,height/2),angle,1)
    mat_rotation[0,2] += (width_new - width)/2
    mat_rotation[1,2] += (height_new - height)/2

    return mat_rotation, width_new, height_new


def rotate_lossless(image, angle):
    height, width = image.shape[:2]
    matrix, width_new, height_new = get_rotate_lossless_matrix(angle, width, height)
    image = cv2.warpAffine(image, matrix,
                           (width_new, height_new),
                           flags=cv2.INTER_LINEAR)

    return image



def blend_matting(background, foreground, mask):
    # mask
    mask = mask.astype(np.float32) / 255
    mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
    height, width = foreground.shape[0:2]
    # fg
    foreground = cv2.multiply(mask, foreground.astype(np.float32))
    # bg
    background = cv2.resize(background, (width, height)).astype(np.float32)
    background = cv2.multiply(1 - mask, background)
    # blend
    new_image = cv2.add(background, foreground)

    return new_image.astype(np.uint8)


def add_edge_in_png(image, edge_side=20, kerner_size=5, iteration=4, color=(255,255,255)):
    # add edge on image
    if len(image.shape) < 3 or image.shape[2] != 4:
        warnings.warn('image shape {}, do nothing'.format(image.shape))
        return image
    mask = image[:, :, 3]
    mask[mask > 0] = 255
    kernel = np.ones((kerner_size, kerner_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=iteration)
    mask[mask > 0] = 255
    contours , hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_edge = cv2.drawContours(image, contours, -1, color, edge_side)
    image_with_edge[:,:,3] = mask
    return image_with_edge


def crop_png_by_alpha(image):
    if len(image.shape) < 3 or image.shape[2] != 4:
        warnings.warn('image shape {}, do nothing'.format(image.shape))
        return image

    mask = image[:, :, 3:]
    if (mask > 0).all():
        return image

    contours , hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.array([])
    for i in range(0, len(contours)):
        pts = contours[i].reshape((contours[i].shape[0], contours[i].shape[2]))
        if i == 0:
            points = pts
        else:
            points = np.concatenate((points, pts),axis=0)
    xmin = min(points[:,0])
    xmax = max(points[:,0])
    ymin = min(points[:,1])
    ymax = max(points[:,1])
    image = image[ymin:ymax, xmin:xmax, :].copy()

    return image


def rotate_crop_blend_png(image, background, angle):
    # 对png图像进行旋转, 剪裁, 粘贴
    if len(image.shape) < 3 or image.shape[2] != 4:
        warnings.warn('image shape {}, do nothing'.format(image.shape))
        return image
    image = rotate_lossless(image, angle)
    image = crop_png_by_alpha(image)
    image = blend_matting(background, image[:,:,:3], image[:,:,3])

    return image





