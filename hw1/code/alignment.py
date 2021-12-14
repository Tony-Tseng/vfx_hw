import cv2
import numpy as np
import os
from typing import List, Tuple


def build_pyramid(img: np.array, level: int) -> List[int]:
    """
    Args:
        img: greyscale image
        level: the times of downsampling
    Returns:
        pyramid: level[0] is the original image, level[n] where n = 1..level is the half size of level[n - 1]
    """
    assert level >= 0 and np.log2(max(img.shape)) >= level

    pyramid = [img]
    h, w = img.shape[0], img.shape[1]
    for i in range(level):
        w, h = max(w // 2, 1), max(h // 2, 1)
        half = cv2.resize(pyramid[i], (w, h))
        pyramid.append(half)

    return pyramid


def shift(img: np.array, offset: np.array) -> np.array:
    """
    Args:
        img: image
        offset: [dr, dc]
            dr: row shift(+ for moving the image downward)
            dc: column shift(+ for moving the image rightward)
    Returns:
        result: shifted image
    """

    [dr, dc] = offset.astype(int)
    pad_r = (dr, 0) if dr > 0 else (0, -dr)
    pad_c = (dc, 0) if dc > 0 else (0, -dc)

    pad_img = np.pad(img, [pad_r, pad_c])

    rb = -min(dr, 0)
    re = rb + img.shape[0]
    cb = -min(dc, 0)
    ce = cb + img.shape[1]

    return pad_img[rb:re, cb:ce]


def gen_mask(img: np.array, threshold: int) -> Tuple[np.array, np.array]:
    """
    Args:
        img: greyscale image [0-127]
        threshold: skipped range
    Returns:
        result: value, mask
    """

    m = np.median(img)
    value = np.zeros_like(img, dtype=int)
    value[img > m] = 1

    mask = np.ones_like(img, dtype=int)
    mask[np.abs(img - m) < threshold] = 0

    return value, mask


def mtb(img_ref: np.array, img_cmp: np.array, threshold: int, max_level: int) -> np.array:
    """
    Args:
        img_ref: greyscale reference image [0-127]
        img_cmp: greyscale translated image [0-127]
        threshold: skipped range
        max_level: the maximum times of downsampling
    Returns:
        result: [dr, dc]
    """

    max_level = min(int(np.log2(max(img_ref.shape))) - 1, max_level)

    refs = build_pyramid(img_ref, max_level)
    cmps = build_pyramid(img_cmp, max_level)        

    offset = np.zeros(2, dtype=int)

    for i in range(max_level, -1, -1):
        offset = offset * 2

        min_error = np.prod(refs[i].shape)
        min_drc = np.zeros(2, dtype=int)

        ref_val, ref_mask = gen_mask(refs[i], threshold)
        cmp_val, cmp_mask = gen_mask(cmps[i], threshold)

        # cv2.imwrite(os.path.join('output_images', 'align', f'ref_{i}.png'), ref_val * 255)
        # cv2.imwrite(os.path.join('output_images', 'align', f'cmp_{i}.png'), cmp_val * 255)

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                drc = np.array([dr, dc])                

                cmp_val_shifted = shift(cmp_val, offset + drc)
                cmp_mask_shifted = shift(cmp_mask, offset + drc)

                diff = np.logical_xor(ref_val, cmp_val_shifted)
                mask = np.logical_and(ref_mask, cmp_mask_shifted)

                error = np.logical_and(diff, mask).sum()

                if error < min_error:
                    min_error = error
                    min_drc = drc

        offset = offset + min_drc

    return offset


def test():
    indir = os.path.join('source_images', 'align')
    outdir = os.path.join('output_images', 'align')

    a = cv2.imread(os.path.join(indir, '01.JPG'))
    b = cv2.imread(os.path.join(indir, '02.JPG'))

    c = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    d = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    ans = mtb(c, d, 3, 3)

    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, '01.jpg'), a)

    nimg = np.zeros_like(b)
    for i in range(b.shape[2]):
        nimg[:, :, i] = shift(b[:, :, i], ans)

    cv2.imwrite(os.path.join(outdir, '02.jpg'), nimg)

    #cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    # for i in range(-5, 6):
    #     for j in range(-5, 6):
    #         res = mtb(shift(a, np.array([i, j], dtype=int)), a, 5, 5)
    #         if (res != [i, j]).any():
    #             print('wtf')

