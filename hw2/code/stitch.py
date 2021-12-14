import numpy as np
import numpy.linalg as LA
import cv2
import sys
from estimator import Estimator
import math

DEBUG = False


def cylindrical_mapping(img, focal):
    res = np.zeros_like(img)

    h, w, _ = img.shape

    xs, ys = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    xs = xs - w / 2
    ys = ys - h / 2

    xs = (focal * np.arctan2(xs, focal))
    ys = (focal * ys / np.sqrt(xs ** 2 + focal ** 2))

    xs = np.round(xs + w / 2).astype(int)
    ys = np.round(ys + h / 2).astype(int)

    res[ys, xs] = img

    return res


def merge(img1, img2, orig_border_r, blend_w):
    res = np.zeros_like(img1).astype(np.float64)

    res[:, :(orig_border_r - blend_w // 2), :] = img1[:, :(orig_border_r - blend_w // 2), :]
    res[:, (orig_border_r + blend_w // 2):, :] = img2[:, (orig_border_r + blend_w // 2):, :]

    mid = range(orig_border_r - blend_w // 2, orig_border_r + blend_w // 2)

    def gen_mask(img):
        mask = np.sum(img, axis=2) > 35
        mask = cv2.erode(mask.astype(np.float32), np.ones(2))
        return mask[:, mid, np.newaxis]

    mask1 = gen_mask(img1)
    mask2 = gen_mask(img2)
    mask = np.logical_and(mask1, mask2)

    print(mask.shape)

    res[:, mid, :] = multi_band_blend(img1[:, mid, :], img2[:, mid, :]) * mask
    # res[:, mid, :] = linear_blend(img1[:, mid, :], img2[:, mid, :]) * mask
    res[:, mid, :] = res[:, mid, :] + img1[:, mid, :] * np.logical_and(mask1, 1 - mask2)
    res[:, mid, :] = res[:, mid, :] + img2[:, mid, :] * np.logical_and(1 - mask1, mask2)

    if DEBUG:
        cv2.imwrite('blend.jpg', res)
        cv2.imwrite('mask.jpg', mask * 255)

    return res


def linear_blend(img1, img2):
    weight = np.linspace(0, 1, num=img1.shape[1]).reshape(1, -1, 1)
    res = img1 * (1 - weight) + img2 * weight
    return res


def multi_band_blend(img1, img2, leveln=5):

    def gaussian(img):
        pyr = [img]
        for i in range(leveln - 1):
            img_half = cv2.pyrDown(pyr[i])
            pyr.append(img_half)
        return pyr

    def laplacian(img):
        pyr = gaussian(img)
        res = []
        for i in range(leveln - 1):
            h, w, _ = pyr[i].shape
            l = pyr[i] - cv2.pyrUp(pyr[i + 1], dstsize=(w, h))
            res.append(l)
        res.append(pyr[-1])
        return res

    blend_w = img1.shape[1]

    img1 = img1.copy().astype(np.float64)
    img2 = img2.copy().astype(np.float64)
    mask = np.zeros_like(img1)
    mask[:, :blend_w // 2, :] = 1
    # mask[:, blend_w // 2, :] = 0.5

    mask_pyr = gaussian(mask)

    pyr1 = laplacian(img1)
    pyr2 = laplacian(img2)

    if DEBUG:
        for i, (a, b) in enumerate(zip(pyr1, pyr2)):
            cv2.imwrite(f'LPA_{i}.jpg', a)
            cv2.imwrite(f'LPB_{i}.jpg', b)

    blend_pyr = []
    for i, v in enumerate(mask_pyr):
        b = pyr1[i] * v + pyr2[i] * (1 - v)
        blend_pyr.append(b)

        if DEBUG:
            cv2.imwrite(f'LPC_{i}.jpg', b)

    res = blend_pyr[-1]
    for v in blend_pyr[-2::-1]:
        h, w, _ = v.shape
        res = cv2.pyrUp(res, dstsize=(w, h))
        res += v

    if DEBUG:
        cv2.imwrite('multi_band_blend.jpg', res)

    return res


def warp(img1, img2, H):

    def affine_transform(matrix, point):
        r = matrix @ np.concatenate((point, np.ones(1)))
        r = r / r[-1]
        return r[:-1]

    left_up = affine_transform(H, np.array([0, 0]))
    left_down = affine_transform(H, np.array([0, img2.shape[0]]))
    right_up = affine_transform(H, np.array([img2.shape[1], 0]))
    right_down = affine_transform(H, np.array([img2.shape[1], img2.shape[0]]))
    corners = np.concatenate((left_up, left_down, right_up, right_down)).reshape(-1, 2)

    print('Corners:', corners)

    if left_up[0] < 0:
        raise ValueError('img2 should be on the right of img1!')

    y_offset = int(abs(corners[:, 1].min()))
    w = int(np.max([img1.shape[1], img1.shape[1], corners[:, 0].max()]))
    h = int(np.max([img1.shape[0], img1.shape[0], corners[:, 1].max()]))

    print(f'(y_offset, w, h) = ({y_offset}, {w}, {h})')

    pad1 = np.pad(img1, [(y_offset, h - img1.shape[0]), (0, w - img1.shape[1]), (0, 0)])
    pad2 = np.pad(img2, [(0, h - img2.shape[0] + y_offset), (0, w - img2.shape[1]), (0, 0)])

    if DEBUG:
        cv2.imwrite('pad1.jpg', pad1)
        cv2.imwrite('pad2.jpg', pad2)

    h, w, _ = pad2.shape
    warp1 = cv2.warpPerspective(pad1, np.eye(3), (w, h))
    warp2 = cv2.warpPerspective(pad2, H, (w, h))
    # warp2 = tx.warp(pad2, H) * 255

    if DEBUG:
        cv2.imwrite('warp1.jpg', warp1)
        cv2.imwrite('warp2.jpg', warp2)

    print(f'border_r = {int((left_up[0] + left_down[0]) / 2)}')

    return warp1, warp2, int((left_up[0] + img1.shape[1]) / 2), int((img1.shape[1] - left_up[0]) / 2) - 10


class Stitch:
    def __init__(self, file_path, w, h):

        self.images = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\r\n') for line in f.readlines()]
            names = lines[0::2]
            focals = lines[1::2]

        print(names)
        print(focals)

        for name, focal in zip(names, focals):
            img = cv2.imread(name)
            img = cv2.resize(img, (w, h))
            img = cylindrical_mapping(img, float(focal))
            self.images.append(img)

    def stitch(self):
        est = Estimator()
        res = self.images[0]

        H_acc = np.eye(3)

        for i in range(1, len(self.images)):
            masked_w = int(self.images[i].shape[1] // 3 * 2)

            temp = res.copy()
            temp[:, :-masked_w, :] = 0

            if DEBUG:
                cv2.imwrite(f'merge_masked{i - 1}.jpg', temp)

            H = est.match(temp, self.images[i])
            print("Homography :", H)
            a, b, r, bw = warp(res, self.images[i], H)
            res = merge(a, b, r, blend_w=bw)

            # cv2.imwrite('temp.jpg', res)

            # H = self.matcher_obj.match(self.images[i - 1], self.images[i], 'right')
            # print("Homography :", H)
            # H_acc = H_acc @ H
            # a, b, r = warp(res, self.images[i], H_acc)

        return res
