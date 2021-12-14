import matplotlib.pyplot as plt
from PIL import Image, ExifTags
import numpy as np
import cv2
import os
import re
from pathlib import Path


def gsolve(Z, B, l, w):
    '''
    Solve the linear solution of Debevec's Method
    Args:
        Z(i,j) is the pixel values of pixel location number i in image j
        B(j) is the log delta t, or log shutter speed for image j
        l: lambda for regularization term of optimization formula
        w: weight of pixel values (0~255)
    Return:
        g: inverse camera response function from (0~255) pixel values.
        lE: log exposure in the real world.
    '''
    image_range = 256
    n, p = Z.shape
    A = np.zeros((n*p+image_range-1, image_range+n))
    b = np.zeros((A.shape[0], 1))

    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, image_range+i] = -wij
            b[k, 0] = wij*B[j]
            k = k+1

    A[k, 127] = 1
    k = k+1

    for i in range(1, 255):
        A[k, i-1:i+2] = l * w[i] * np.array([1, -2, 1])
        k = k+1

    x = np.linalg.lstsq(A, b, rcond=None)[0]

    g = x[:image_range]
    lE = x[image_range:]
    return g, lE


def draw_response_curve(g_list, lE_list, Z, B, filename):
    '''
    Draw the respones curve.
    Args:
        g_list: inverse CRF of different channel
        lE_list: log exposure of different channel
        Z(i,j): sample points of i position and j shutter speed
        B: log shutter speed
    '''
    color = ['red', 'green', 'blue']
    x = np.arange(256)

    # for channel,lE in enumerate(lE_list):
    #     for j in range(Z.shape[1]):
    #         plt.scatter(-lE-B[j], Z[:,j], s=10)
    #     plt.title(f"{color[channel]} channel")
    #     plt.xlabel("log Exposure")
    #     plt.ylabel("pixel value")
    #     plt.show()

    for channel, g in enumerate(g_list):
        plt.plot(g, x, color=color[channel])
    plt.title("Camera Response Function")
    plt.xlabel("log Exposure")
    plt.ylabel("pixel value")
    plt.savefig(filename)
    plt.show(block=False)
    plt.close()


def construct_radiance(img_dict, w, g, delta_t, channel):
    '''
    Calculate and average the radiance of the real world radiance 
    according to the set of input images.
    Args:
        img_data: image data of one channel of all input images
        w: the weighting function
        g: inverse camera response function from (0~255) pixel values.
        delta_t: log shutter speed
        channel: channel of images
    Returns:
        rad: return the average real world radiance distribution
    '''
    exposure = []
    img_data = []

    for i, img_name in enumerate(img_dict):
        img = img_dict[f"{img_name}"]["data"][:, :, channel]
        img_data.append(img)
        exposure.append(g[img] - delta_t[i])

    img_array = np.array(img_data)
    rad = np.average(exposure, axis=0, weights=w[img_array])

    return rad


def draw_radiance_map(rad_list, hdr_name, log_name = None):
    '''
    Draw the relative radiance map of three channel R, G, B, and save as hdr file.
    Arg:
        rad_list: radiance map of R, G, B
        file_name: the file name of hdr image
    '''
    hdr_new = np.transpose(np.exp(np.squeeze(np.array(rad_list))), (1, 2, 0)).astype(np.float32)
    cv2.imwrite(f'{hdr_name}', hdr_new)

    gray = np.log2(cv2.cvtColor(hdr_new, cv2.COLOR_BGR2GRAY))
    plt.imshow(gray, cmap=plt.cm.jet)
    plt.colorbar()

    if log_name is not None:
        plt.savefig(f'{log_name}')
        
    plt.show(block=False)
    plt.close()

    # for rad in rad_list:
    #     plt.imshow(rad, cmap=plt.cm.jet)
    #     plt.colorbar()
    #     plt.show()

    return hdr_new

    