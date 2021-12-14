#  Harris detector
import scipy.ndimage.filters as nd_filters
import numpy as np
# from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import cv2

class Harris:
    def __init__(self, image, sigma=2, diff_filter_name='origin'):
        self.image = image
        self.sigma = sigma
        self.gray = np.array(self.image,dtype=float).dot(np.array([0.299,0.587,0.114]))
        self.diff_filter_name = diff_filter_name
        self.height, self.width,_ = image.shape

    def calculate_gradient(self, diff_filter, pre_filter):
        diff_x = diff_filter['dx']
        diff_y = diff_filter['dy']
        Ix = -signal.convolve2d(self.gray, diff_x, 'same')
        Iy = -signal.convolve2d(self.gray, diff_y, 'same')

        g = pre_filter
        Ix2 = signal.convolve2d(Ix*Ix, g, 'same')
        Iy2 = signal.convolve2d(Iy*Iy, g, 'same')
        Ixy = signal.convolve2d(Ix*Iy, g, 'same')

        I_dict = {}
        I_dict['Ix2'] = Ix2
        I_dict['Iy2'] = Iy2
        I_dict['Ixy'] = Ixy
        I_dict['Ix'] = Ix
        I_dict['Iy'] = Iy

        return I_dict

    def corner_feature(self, I_dict, threshold=2, alpha=0.04, r=6):
        Ix2 = I_dict['Ix2']
        Iy2 = I_dict['Iy2']
        Ixy = I_dict['Ixy']

        R = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                M = np.array([[Ix2[i,j], Ixy[i,j]],[Ixy[i,j], Iy2[i,j]]])
                R[i,j] = np.linalg.det(M) - alpha*np.trace(M)**2
        R=(1000/np.max(R))*R

        # check R value
        r_arr = np.zeros_like(R)
        for r_idx, r in enumerate(R):
            for c_idx, r in enumerate(r):
                if (r > threshold):
                    # this is a corner
                    r_arr[r_idx, c_idx] = r
        # Local maximum
        corners = peak_local_max(r_arr, min_distance=13)
        new_corners = np.zeros_like(corners)
        new_corners[:,0] = corners[:,1]
        new_corners[:,1] = corners[:,0]

        return new_corners

    def get_diff_filter(self):
        # horizontal gradient filter
        name = self.diff_filter_name

        diff_filter = {}
        if(name=='origin'):
            diff_filter['dx'] = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            diff_filter['dy'] = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        elif(name=='sobel'):
            diff_filter['dx'] = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            diff_filter['dy'] = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).T
        else:
            print("filter type not defined!")
        return diff_filter

    def get_orientations(self, I_dict, bins=8, ksize=9):
        Ix2 = I_dict['Ix2']
        Iy2 = I_dict['Iy2']
        Ix = I_dict['Ix']
        Iy = I_dict['Iy']

        M = (Ix2 + Iy2) ** (1/2)

        theta = np.arctan(Iy / (Ix + 1e-8)) * (180 / np.pi)
        theta[Ix < 0] += 180
        theta = (theta + 360) % 360

        bin_size = 360. / bins
        theta_bins = (theta + (bin_size / 2)) // int(bin_size) % bins # divide to 8 bins

        ori_1hot = np.zeros((bins,) + Ix.shape)
        for b in range(bins):
            ori_1hot[b][theta_bins == b] = 1
            ori_1hot[b] *= M
            ori_1hot[b] = cv2.GaussianBlur(ori_1hot[b], (ksize, ksize), 0)

        ori = np.argmax(ori_1hot, axis=0)

        return ori, ori_1hot, theta, theta_bins, M

    def get_descriptors(self, fpx, fpy, ori, theta):
        
        bins, h, w = ori.shape

        def get_sub_vector(fy, fx, oy, ox, ori):
            sv = []
            for b in range(bins):
                sv.append(np.sum(ori[b][fy:fy+oy, fx:fx+ox]))
                
            sv_n1 = [x / (np.sum(sv) + 1e-8) for x in sv]
            sv_clip = [x if x < 0.2 else 0.2 for x in sv_n1]
            sv_n2 = [x / (np.sum(sv_clip) + 1e-8) for x in sv_clip]
            
            return sv_n2
        
        def get_vector(y, x):
            # +angle in cv2 is counter-clockwise.
            # +y is down in image coordinates.
            M = cv2.getRotationMatrix2D((12, 12), theta[y, x], 1)
            if y-12 < 0 or x-12 < 0: return 0
            ori_rotated = [cv2.warpAffine(t[y-12:y+12, x-12:x+12], M, (24, 24)) for t in ori]
            
            vector = []
            subpatch_offsets = [4, 8, 12, 16]
            for fy in subpatch_offsets:
                for fx in subpatch_offsets:
                    vector += get_sub_vector(fy, fx, 4, 4, ori_rotated)
                    
            return vector

        descriptors = []
        for y, x in zip(fpy, fpx):
            vector = get_vector(y, x)
            if np.sum(vector) > 0:
                descriptors.append(vector)
        
        print('descriptors: %d' % len(descriptors))
        descriptors = np.array(descriptors)
        return descriptors

    def get_keypoint(self):
        filter_size = 2 * 6 * self.sigma

        pre_filter = gaussian_filter(self.sigma, filter_size)
        diff_filter = self.get_diff_filter()
        
        I_dict = self.calculate_gradient(diff_filter, pre_filter)
        keypoint = self.corner_feature(I_dict, threshold=10, alpha=0.04, r=6)

        ori, ori_1hot, theta, theta_bins, M = self.get_orientations(I_dict)
        des = self.get_descriptors(keypoint[:,0], keypoint[:,1], ori_1hot, theta)

        return keypoint, des

    def plot_feature(self, keypoint):
        x = [kp[0] for kp in keypoint]
        y = [kp[1] for kp in keypoint]

        plt.figure()
        fig = plt.imshow(self.image)
        plt.scatter(x,y, facecolors='none', edgecolors='r') #,s=3
        plt.show()

def gaussian_filter(sigma, filter_size=24):
    nx, ny = (filter_size, filter_size)
    x = np.linspace(-nx/2, nx/2, nx)
    y = np.linspace(-ny/2, ny/2, ny)
    xv, yv = np.meshgrid(x, y)
    gaussian_filter = np.exp(-(xv*xv+yv*yv)/(2*sigma**2))

    return gaussian_filter

if __name__=='__main__':
    im_dir = '/Users/tsenghungyen/Desktop/'
    im_file = "lib1.jpg"

    image = cv2.imread(im_dir+im_file)
    image_harris = Harris(image, diff_filter_name='origin')

    keypoint, feat = image_harris.get_keypoint()
    image_harris.plot_feature(keypoint)