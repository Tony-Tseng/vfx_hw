from tone_mapping import global_tone_mapping, local_tone_mapping, gaussian, joint_bilateral, reinhard
from alignment import mtb, shift, test
from hdr import *
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
import numpy as np
import cv2
import os
import re
from pathlib import Path
# import warnings
# warnings.filterwarnings("ignore")


def store_image_info(shutter_file, dir_path, image_name_pattern, img_list):
    '''
    Args:
        shutter_file: txt file
        dir_path: directory of images
        image_name_pattern: the prefix of image name
        img_list: the name of images
    Return:
        image_dict: A dictionary of dictionary that use image name as key and (data and shutter) as value
        h: height of the images
        w: width of the images
        c: channel of the images
    '''
    image_dict = {}
    for i, img_name in enumerate(img_list):
        dict_img_name = Path(img_name).stem
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        image_dict[f"{dict_img_name}"] = {}
        image_dict[f"{dict_img_name}"]['data'] = img
    h, w, c = img.shape

    with open(os.path.join(dir_path, shutter_file), encoding='utf8', newline='\r\n') as f:
        for line in f:
            split_info = line.split()
            if(re.match(image_name_pattern, split_info[0])):
                image_dict[Path(split_info[0]).stem]["shutter"] = 1 / float(split_info[1])
    return image_dict, h, w, c


def store_image_info_from_photo(dir_path, image_name_pattern, img_list):
    image_dict = {}
    for i, img_name in enumerate(img_list):
        dict_img_name = Path(img_name).stem
        img_path = os.path.join(dir_path, img_name)

        image_dict[f"{dict_img_name}"] = {}
        img = Image.open(img_path)
        img_exif = img.getexif()
        for tag_id in img_exif:
            # get the tag name, instead of human unreadable tag id
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if(tag == 'ExposureTime'):
                data = img_exif.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode()
                image_dict[f"{dict_img_name}"]['shutter'] = float(data)
            # print(f"{tag:25}: {data}")
        img = cv2.imread(img_path)
        image_dict[f"{dict_img_name}"]['data'] = img

    h, w, c = img.shape

    return image_dict, h, w, c


def load_memorial():
    path = os.path.join('source_images', 'memorial')
    img_list = [fn for fn in os.listdir(path) if fn.endswith('png')]

    d, h, w, c = store_image_info('memorial.hdr_image_list.txt', path, 'memorial*', img_list)

    return d, h, w, c, 'memorial'


def load_custom(path=None):
    path = path or os.path.join('source_images', 'building')
    img_list = [fn for fn in os.listdir(path) if fn.endswith('JPG')]

    d, h, w, c = store_image_info_from_photo(path, 'DSC*', img_list)
    align(d)

    return d, h, w, c, 'building'


def random_sampling(n_points, img_dict, channel):
    h, w, _ = next(iter(img_dict.values()))['data'].shape

    r = np.random.randint(50, h - 50, size=n_points)
    c = np.random.randint(50, w - 50, size=n_points)

    Z = np.zeros((n_points, len(img_dict)), dtype=int)
    B = np.zeros((len(img_dict), 1), dtype=float)

    for i, (k, v) in enumerate(img_dict.items()):
        Z[:, i] = v['data'][r, c, channel]
        B[i] = np.log(v['shutter'])

    return Z, B


def evenly_spaced_sampling(n_points, img_dict, channel):
    h, w, _ = next(iter(img_dict.values()))['data'].shape

    scale = np.sqrt(n_points / w / h)
    nr = int(np.round(h * scale))
    nc = int(np.round(w * scale))

    r = np.linspace(50, h - 50, nr + 2)[1:-1].astype(int)
    c = np.linspace(50, w - 50, nc + 2)[1:-1].astype(int)

    Z = np.zeros((nr * nc, len(img_dict)), dtype=int)
    B = np.zeros((len(img_dict), 1), dtype=float)

    for i, (k, v) in enumerate(img_dict.items()):
        Z[:, i] = v['data'][r, :, channel][:, c].flatten()
        B[i] = np.log(v['shutter'])

    return Z, B


def align(image_dict):
    ref = [v for i, v in enumerate(
        image_dict.values()) if i == len(image_dict) // 2]
    ref = cv2.cvtColor(ref[0]['data'], cv2.COLOR_BGR2GRAY)

    offsets = []
    for i, k in enumerate(image_dict):
        print(f'{i + 1} / {len(image_dict)}')
        img = cv2.cvtColor(image_dict[k]['data'], cv2.COLOR_BGR2GRAY)
        offset = mtb(ref, img, 3, 3)
        offsets.append(offset)

    for i, k in enumerate(image_dict):
        img = image_dict[k]['data']
        nimg = np.zeros_like(img)
        for c in range(img.shape[2]):
            nimg[:, :, c] = shift(img[:, :, c], offsets[i])

        image_dict[k]['data'] = nimg


def exp(custom=False, rnd_sampling=False, l=50, w_step=False, n_points=100):
    if not custom:
        image_dict, h, w, c, name = load_memorial()
    else:
        image_dict, h, w, c, name = load_custom()

    dir = os.path.join('output_images', f'{name}_{"rnd" if rnd_sampling else "even"}_{l}_{"step" if w_step else "one"}_{n_points}')
    os.makedirs(dir, exist_ok=True)

    g_list = []
    lE_list = []
    rad_list = []

    if w_step:
        a = np.linspace(0, 1, 128)
        b = np.concatenate((a, np.flip(a)))
        w = b.reshape(-1, 1)
    else:
        w = np.ones((256, 1))

    for channel in range(c):
        if rnd_sampling:
            Z, B = random_sampling(n_points, image_dict, channel)
        else:
            Z, B = evenly_spaced_sampling(n_points, image_dict, channel)

        g, lE = gsolve(Z, B, l, w)
        rad = construct_radiance(image_dict, w, g, B, channel)

        g_list.append(g)
        lE_list.append(lE)
        rad_list.append(rad)

    draw_response_curve(g_list, lE_list, Z, B, os.path.join(dir, 'response_curve.jpg'))
    HDRIMG = draw_radiance_map(rad_list, os.path.join(dir, 'result.hdr'), os.path.join(dir, 'log.jpg')).astype(np.float32)

    img_gaussian = local_tone_mapping(HDRIMG, Filter=joint_bilateral, window_size=39, sigma_s=100, sigma_r=0.8)
    cv2.imwrite(os.path.join(dir, 'joint_bilateral.jpg'), img_gaussian)


def gen_pic():
    os.chdir('..')

    # (1)
    exp(l=10, rnd_sampling=True)

    # (2): sampling difference to (1)
    exp(l=10, rnd_sampling=False)

    # (3): lambda difference to (2)
    exp(l=50, rnd_sampling=False)

    # (4): w difference to (1)
    exp(l=10, rnd_sampling=True, w_step=True)

    # (5): sampling isn't an important factor
    exp(l=10, rnd_sampling=True, n_points=500)
    exp(l=10, rnd_sampling=False, n_points=500)

    # artifact
    exp(custom=True, w_step=True)
    exp(custom=True, w_step=True, l=100)

    # mtb test
    test()

    # reinhard (1)
    dir = os.path.join('output_images', 'building_even_100_step_100')
    img = cv2.imread(os.path.join(dir, 'result.hdr'), -1)
    r = reinhard(img, 6, 0.5, 1, 0)
    cv2.imwrite(os.path.join(dir, 'reinhard.png'), r * 255)

    # reinhard (2)
    dir = os.path.join('output_images', 'building_even_50_step_100')
    img = cv2.imread(os.path.join(dir, 'result.hdr'), -1)
    r = reinhard(img, 6, 0.5, 1, 0)
    cv2.imwrite(os.path.join(dir, 'reinhard.png'), r * 255)

    # reinhard (3)
    dir = os.path.join('output_images', 'memorial_rnd_10_one_100')
    img = cv2.imread(os.path.join(dir, 'result.hdr'), -1)
    r = reinhard(img, 6, 0.5, 1, 0)
    cv2.imwrite(os.path.join(dir, 'reinhard.png'), r * 255)


def gen_data(dir):
    image_dict, h, w, c, name = load_custom(dir)

    g_list = []
    lE_list = []
    rad_list = []

    a = np.linspace(0, 1, 128)
    b = np.concatenate((a, np.flip(a)))
    w = b.reshape(-1, 1)
    l = 100

    for channel in range(c):
        Z, B = evenly_spaced_sampling(100, image_dict, channel)

        g, lE = gsolve(Z, B, l, w)
        rad = construct_radiance(image_dict, w, g, B, channel)

        g_list.append(g)
        lE_list.append(lE)
        rad_list.append(rad)

    HDRIMG = draw_radiance_map(rad_list, os.path.join(dir, 'result.hdr')).astype(np.float32)

    r = reinhard(HDRIMG, 6, 0.5, 1, 0)
    cv2.imwrite(os.path.join(dir, 'reinhard.jpg'), r * 255)
    
    img_gaussian = local_tone_mapping(HDRIMG, Filter=joint_bilateral, window_size=39, sigma_s=100, sigma_r=0.8)
    cv2.imwrite(os.path.join(dir, 'joint_bilateral.jpg'), img_gaussian)



if __name__ == "__main__":
    gen_data(os.path.join('..', 'data'))
    #gen_pic()
