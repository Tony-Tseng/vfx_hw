import numpy as np
import cv2

def reinhard(hdr_img, intensity, contrast, adaption, color_correction):
    img = (hdr_img - hdr_img.min()) / (hdr_img.max() - hdr_img.min())
    result = np.zeros_like(img)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey[grey == 0] = 1e-4
    lgrey = np.log(grey)

    lmean = grey.mean()
    channel_mean = img.mean(axis=(0, 1), dtype=np.float)
    llmin = lgrey.min()
    llmax = lgrey.max()
    llmean = lgrey.mean()
    intensity = np.exp(-intensity)
    contrast = contrast if contrast > 0 else 0.3 + 0.7 * np.power((llmax - llmean) / (llmax - llmin), 1.4)


    for c in range(hdr_img.shape[2]):
        I_l = color_correction * img[:, :, c] + (1 - color_correction) * grey
        I_g = color_correction * channel_mean[c] + (1 - color_correction) * lmean
        I_a = adaption * I_l + (1 - adaption) * I_g

        result[:, :, c] = img[:, :, c] / (img[:, :, c] + np.power(intensity * I_a, contrast))
    
    return result

def global_tone_mapping(HDRIMG, WB = 'True'):
    """ Perform Global tone mapping on HDRIMG
            Note:
                1.Please remember clip the range of intensity to [0, 1] and convert type of LDRIMG to "uint8" with range [0, 255] for display. You can use the following code snippet.
                  >> LDRIMG = np.round(LDRIMG*255).astype("uint8")
                2.Make sure the LDRIMG's range is in 0-255(uint8). If the value is larger than upperbound, modify it to upperbound. If the value is smaller than lowerbound, modify it to lowerbound.
                3.Beware on the log results of 0(small number). Replace the small numbers(less then DBL_MIN) with DBL_MIN. The following code shows how to get DBL_MIN in python.
                  >> import sys
                  >> sys.float_info.min # get DBL_MIN
            Args:
                HDRIMG (np.ndarray): The input image to process
                WB (bool): The flag to indicate whether to perform white balance
            Returns:
                LDRIMG (np.ndarray): The processed corresponding low dynamic range image of HDRIMG
            Todo:
                - implement global tone mapping (with white balance) here
    """
    if WB == 'True':
      HDRIMG = white_balance(HDRIMG,x_range=(457,481),y_range=(400,412))

    LDRIMG = np.zeros_like(HDRIMG)
    X0 = np.zeros_like(HDRIMG)
    X0[:,:,0] = np.max(HDRIMG[:,:,0])
    X0[:,:,1] = np.max(HDRIMG[:,:,1])
    X0[:,:,2] = np.max(HDRIMG[:,:,2])
    
    s = 0.9
    LDRIMG = 2 ** (s * (np.log2(HDRIMG) - np.log2(X0)) + np.log2(X0))

    gamma = 1/2.2
    LDRIMG = LDRIMG ** gamma
    LDRIMG[LDRIMG>1] = 1
    LDRIMG[LDRIMG<0] = 0
    LDRIMG = np.round(LDRIMG*255).astype("uint8")

    return LDRIMG

 
def local_tone_mapping(HDRIMG, Filter, window_size, sigma_s, sigma_r):
    """ Perform Local tone mapping on HDRIMG
            Note:
                1.Please remember clip the range of intensity to [0, 1] and convert type of LDRIMG to "uint8" with range [0, 255] for display. You can use the following code snippet.
                  >> LDRIMG = np.round(LDRIMG*255).astype("uint8")
                2.Make sure the LDRIMG's range is in 0-255(uint8). If the value is larger than upperbound, modify it to upperbound. If the value is smaller than lowerbound, modify it to lowerbound.
            Args:
                HDRIMG (np.ndarray): The input image to process
                Filter (function): 'Filter' is a function that is used for filter operation to get base layer. It can be gaussian or bilateral. 
                                   It's input is log of the intensity and filter's parameters. And the output is the base layer.
                window size(diameter) (int): default 35
                sigma_s (int): default 100
                sigma_r (float): default 0.8
            Returns:
                LDRIMG (np.ndarray): The processed corresponding low dynamic range image of HDRIMG
            Todo:
                - implement local tone mapping here
    """
    LDRIMG = np.zeros_like(HDRIMG)
    I = np.average(HDRIMG,axis=2)
    Cr = HDRIMG[:,:,0]/I
    Cg = HDRIMG[:,:,1]/I
    Cb = HDRIMG[:,:,2]/I

    L = np.log2(I)
    LB = Filter(HDRIMG,window_size, sigma_s, sigma_r) 
    LD = L - LB

    Lmin = np.min(LB)
    Lmax = np.max(LB)

    scale = 3
    LB_ = scale * (LB-Lmax) / (Lmax-Lmin)

    I_ = 2 ** (LB_+LD)

    LDRIMG[:,:,0] = Cr * I_
    LDRIMG[:,:,1] = Cg * I_
    LDRIMG[:,:,2] = Cb * I_

    gamma = 1/2.2
    LDRIMG = LDRIMG ** gamma

    LDRIMG[LDRIMG>1] = 1
    LDRIMG[LDRIMG<0] = 0
    LDRIMG = np.round(LDRIMG*255).astype("uint8")
    return LDRIMG


def gaussian(HDRIMG,window_size,sigma_s,sigma_r):
    """ Perform gaussian filter 
            Notes:
                Please use "symmetric padding" for image padding
            Args:
                HDRIMG: HDR image 
                window size(diameter) (int): default 39
                sigma_s (int): default 100
                sigma_r (float): default 0.8
            Returns:
                LB (np.ndarray): The base layer
            Todo:
                - implement gaussian filter for local tone mapping
    """
    
    I = np.average(HDRIMG,axis=2)
    L = np.log2(I)
    
    #version1 - 2x faster
    # from scipy import signal
    # LB = np.zeros_like(L)
    # one = np.ones_like(L)
    # pad_width = (window_size-1)/2
    
    # x, y = np.meshgrid(np.linspace(-pad_width,pad_width,window_size), np.linspace(-pad_width,pad_width,window_size))
    # gaussian_filter =  np.exp(-(x*x+y*y)/(2*sigma_s**2))

    # one_conv = signal.convolve2d(one, gaussian_filter, boundary='symm', mode='same')
    # LB = signal.convolve2d(L, gaussian_filter, boundary='symm', mode='same')/one_conv

    # version 2 - slower
    LB = np.zeros_like(L)
    pad_width = (window_size-1)//2
    L_pad = np.pad(L,pad_width,'symmetric')

    x, y = np.meshgrid(np.linspace(-pad_width,pad_width,window_size), np.linspace(-pad_width,pad_width,window_size))
    gaussian_filter =  np.exp(-(x*x+y*y)/(2*sigma_s**2))

    for i in range(0,L.shape[0]):
        for j in range(0,L.shape[1]):
            LB[i,j] = np.sum((L_pad[i:i+window_size,j:j+window_size]*gaussian_filter))/np.sum(gaussian_filter)
    
    return LB

def joint_bilateral(HDRIMG,window_size,sigma_s,sigma_r):
    """ Perform bilateral filter 
            Notes:
                Please use "symmetric padding" for image padding   
            Args:
                HDRIMG: HDR image 
                window size(diameter) (int): default 39
                sigma_s (int): default 100
                sigma_r (float): default 0.8
            Returns:
                LB (np.ndarray): The base layer
            Todo:
                - implement bilateral filter for local tone mapping
    """
    I = np.average(HDRIMG,axis=2)
    L = np.log2(I)
    
    LB = np.zeros_like(L)
    pad_width = (window_size-1)//2
    Gray_IMG = np.dot(HDRIMG,[0.2989,0.5870,0.1140])
    L_gray = np.log2(Gray_IMG)
    
    x, y = np.meshgrid(np.linspace(-pad_width,pad_width,window_size), np.linspace(-pad_width,pad_width,window_size))
    gaussian_filter =  np.exp(-(x*x+y*y)/(2*sigma_s**2))

    L_pad = np.pad(L_gray,pad_width,'symmetric')
    L_Gray_pad = np.pad(L_gray,pad_width,'symmetric')

    bilateral_filter = np.zeros((window_size,window_size))
    L_gray_difference = np.zeros((window_size,window_size))

    for i in range(0,L.shape[0]):
        for j in range(0,L.shape[1]):
            L_gray_difference = (L_Gray_pad[i:i+window_size,j:j+window_size]-L_Gray_pad[i+pad_width,j+pad_width])
            bilateral_filter = np.exp( - (L_gray_difference**2) / ( 2*sigma_r**2) ) * gaussian_filter
            LB[i,j] = np.sum((L_pad[i:i+window_size,j:j+window_size]*bilateral_filter))/np.sum(bilateral_filter)
    
    return LB