from harris import gaussian_filter
from scipy import signal
import numpy as np
import math
import csv
# import cv2

class Harris_description:
    def __init__(self, image, keypoint):
        # super().__init__(image)
        self.keypoint = keypoint
        ## If don't inheritance
        self.image = image
        self.gray = np.array(self.image,dtype=float).dot(np.array([0.299,0.587,0.114]))
        self.height, self.width = self.gray.shape

    def cal_magnitude_angle(self):
        g = gaussian_filter(sigma=2)
        L = signal.convolve2d(self.gray, g, 'same')

        magnitude = np.zeros((self.height, self.width))
        angle = np.zeros((self.height, self.width))
        
        for row in range(1,self.height-1):
            for col in range(1,self.width-1):
                magnitude[row,col] = math.sqrt( (L[row+1,col] - L[row-1,col])**2 + (L[row,col+1] - L[row,col-1])**2)
                tmp_angle = (math.atan( (L[row,col+1] - L[row,col-1])/(L[row+1,col] - L[row-1,col]+1e-10) ) )*180 / math.pi
                if(L[row+1,col]-L[row-1,col] <0):
                    angle[row,col] = tmp_angle + 180
                elif(L[row,col+1] - L[row,col-1] < 0):
                    angle[row,col] = tmp_angle + 360
                else:
                    angle[row,col] = tmp_angle
        # print(np.max(angle))
        # print(np.min(angle))
        # cv2.imshow('magnitude',magnitude /np.max(L) )
        # cv2.waitKey(0)
        # cv2.imshow('angle',angle)
        # cv2.waitKey(0)
        return magnitude, angle

    def quantize_orientation(self, theta, num_bins=36):
        bin_width = 360//num_bins
        theta = theta % 360
        return int(np.floor(theta)//bin_width)

    def fit_parabola(self, hist, binno, bin_width): 
        centerval = binno*bin_width + bin_width/2
        if(binno == len(hist)-1):
            rightval = 360 + bin_width/2
        else: 
            rightval = (binno+1)*bin_width + bin_width/2.
        if(binno == 0):
            leftval = -bin_width/2
        else: 
            leftval = (binno-1)*bin_width + bin_width/2

        A = np.array([
            [centerval**2, centerval, 1],
            [rightval**2, rightval, 1],
            [leftval**2, leftval, 1]])
        b = np.array([ hist[binno], 
            hist[(binno+1)%len(hist)], 
            hist[(binno-1)%len(hist)]]) 
        x = np.linalg.lstsq(A, b, rcond=None)[0]

        if(x[0] == 0):
            x[0] = 1e-6

        return -x[1]/(2*x[0])

    def assign_orientation(self, num_bins=36): 
        new_kps = [] 
        bin_width = 360//num_bins

        for kp in self.keypoint: 
            cx, cy = int(kp[0]), int(kp[1])
            # s = np.clip(s, 0, self.gray.shape[2]-1) 
            sigma = 1.5 
            w = int(2*np.ceil(sigma)+1) 
            kernel = gaussian_filter(sigma) 
            hist = np.zeros(num_bins, dtype=np.float32) 
            for oy in range(-w, w+1): 
                for ox in range(-w, w+1): 
                    x, y = cx+ox, cy+oy 
                    if x < 0 or x > self.gray.shape[1]-1: continue 
                    elif y < 0 or y > self.gray.shape[0]-1: continue 
                    m, theta = self.get_grad(self.gray, x, y) 
                    weight = kernel[oy+w, ox+w] * m
                    bin = self.quantize_orientation(theta, num_bins) 
                    hist[bin] += weight
            max_bin = np.argmax(hist) 
            new_kps.append([kp[0], kp[1], kp[2], self.fit_parabola(hist, max_bin, bin_width)]) 
            max_val = np.max(hist) 
            for binno, val in enumerate(hist): 
                if (binno == max_bin): continue 
                if (max_val*0.8 <= val): 
                    new_kps.append([kp[0], kp[1], kp[2], self.fit_parabola(hist, binno, bin_width)])
        return np.array(new_kps)

    # def orientation_assignment(self, filter_size=16, num_bins=36):
    #     new_kp = []
    #     bin_width = 360//num_bins

    #     magnitude, angle = self.cal_magnitude_angle()
    #     w = filter_size//2
    #     hist = np.zeros(num_bins, dtype=np.float32)

    #     for kp in self.keypoint:
    #         row, col, key_val = int(kp[0]), int(kp[1]), float(kp[2])
    #         sigma = 1.5
    #         g = gaussian_filter(sigma=sigma)
    #         for ox in range(-w,w):
    #             for oy in range(-w, w):
    #                 x, y = row+ox, col+oy
    #                 if(x<0 or x>self.gray.shape[1]-1): continue
    #                 elif(y<0 or y>self.gray.shape[0]-1): continue

    #                 weight = g[oy+w,ox+w] * magnitude[y,x]
    #                 bin = self.quantize_orientation(angle[y,x], num_bins)
    #                 hist[bin] += weight
            
    #         max_bin = np.argmax(hist)
    #         new_kp.append([kp[0], kp[1], kp[2], self.fit_parabola(hist, max_bin, bin_width)])
    #         max_val = np.max(hist)

    #         for binno, val in enumerate(hist): 
    #             if binno == max_bin: continue 
    #             if(max_val*0.8) <= val: 
    #                 new_kp.append([kp[0], kp[1], kp[2], self.fit_parabola(hist, binno, bin_width)])

    #     return np.array(new_kp)

    def get_grad(self, L, x, y): 
        dy = L[min(L.shape[0]-1, y+1),x] - L[max(0, y-1),x] 
        dx = L[y,min(L.shape[1]-1, x+1)] - L[y,max(0, x-1)] 
        return self.cart_to_polar_grad(dx, dy)

    def orient_histogram(self, weighted_magnitude, angle_block, num_bins=36):
        feature = []
        hist = np.zeros(num_bins, dtype=np.float32)
        for i in range(16):
            for j in range(16):
                bin = quantize_orientation(angle_block[i,j])
                hist[bin] += weighted_magnitude[i,j]
        
        max_bin = np.argmax(hist)
        return feature

    def get_patch_grads(self, p): 
        r1 = np.zeros_like(p) 
        r1[-1] = p[-1] 
        r1[:-1] = p[1:] 
        r2 = np.zeros_like(p) 
        r2[0] = p[0] 
        r2[1:] = p[:-1] 
        dy = r1-r2 
        r1[:,-1] = p[:,-1] 
        r1[:,:-1] = p[:,1:] 
        r2[:,0] = p[:,0] 
        r2[:,1:] = p[:,:-1] 
        dx = r1-r2 
        return dx, dy

    def get_histogram_for_subregion(self, m, theta, num_bin, reference_angle, bin_width, subregion_w): 
        hist = np.zeros(num_bin, dtype=np.float32) 
        c = subregion_w/2 - .5
        for i, (mag, angle) in enumerate(zip(m, theta)):
            angle = (angle-reference_angle) % 360        
            binno = self.quantize_orientation(angle, num_bin)        
            vote = mag      
        
            hist_interp_weight = 1 - abs(angle - (binno*bin_width + bin_width/2))/(bin_width/2)        
            vote *= max(hist_interp_weight, 1e-6)         
            gy, gx = np.unravel_index(i, (subregion_w, subregion_w))
            x_interp_weight = max(1 - abs(gx - c)/c, 1e-6)            
            y_interp_weight = max(1 - abs(gy - c)/c, 1e-6)        
            vote *= x_interp_weight * y_interp_weight         
            hist[binno] += vote
            hist /= max(1e-6, np.linalg.norm(hist)) 
            hist[hist>0.2] = 0.2 
            hist /= max(1e-6, np.linalg.norm(hist))
        return hist

    def cart_to_polar_grad(self, dx, dy): 
        m = np.sqrt(dx**2 + dy**2) 
        theta = (np.arctan2(dy, dx)+np.pi) * 180/np.pi
        return m, theta 

    def get_local_descriptors(self, kps, w=16, num_subregion=4, num_bin=8):
        descs = []
        bin_width = 360//num_bin

        for kp in kps:
            cx, cy = int(kp[0]), int(kp[1])
            kernel = gaussian_filter(sigma=w) # gaussian_filter multiplies sigma by 3

            t, l = max(0, cy-w//2), max(0, cx-w//2)
            b, r = min(self.gray.shape[0], cy+w//2), min(self.gray.shape[1], cx+w//2)

            patch = self.gray[t:b, l:r]
            dx, dy = self.get_patch_grads(patch)

            if dx.shape[0] < w+1:
                if t == 0: kernel = kernel[kernel.shape[0]-dx.shape[0]:] 
                else: kernel = kernel[:dx.shape[0]]
            if dx.shape[1] < w+1:
                if l == 0: kernel = kernel[:,kernel.shape[1]-dx.shape[1]:] 
                else: kernel = kernel[:,:dx.shape[1]]
            if dy.shape[0] < w+1: 
                if t == 0: kernel = kernel[kernel.shape[0]-dy.shape[0]:] 
                else: kernel = kernel[:dy.shape[0]] 
            if dy.shape[1] < w+1: 
                if l == 0: kernel = kernel[:,kernel.shape[1]-dy.shape[1]:] 
                else: kernel = kernel[:,:dy.shape[1]]

            m, theta = self.cart_to_polar_grad(dx, dy)
            dx, dy = dx*kernel, dy*kernel
            subregion_w = w//num_subregion 
            featvec = np.zeros(num_bin * num_subregion**2, dtype=np.float32) 
            for i in range(0, subregion_w): 
                for j in range(0, subregion_w): 
                    t, l = i*subregion_w, j*subregion_w
                    b, r = min(self.gray.shape[0], (i+1)*subregion_w), min(self.gray.shape[1], (j+1)*subregion_w) 
                    hist = self.get_histogram_for_subregion(m[t:b, l:r].ravel(), theta[t:b, l:r].ravel(), num_bin, float(kp[3]), bin_width, subregion_w) 
                    featvec[i*subregion_w*num_bin + j*num_bin:i*subregion_w*num_bin + (j+1)*num_bin] = hist.flatten() 
            featvec /= max(1e-6, np.linalg.norm(featvec))        
            featvec[featvec>0.2] = 0.2        
            featvec /= max(1e-6, np.linalg.norm(featvec))    
            descs.append(featvec)
        return np.array(descs)

    def get_features(self):
        new_kp = self.assign_orientation()
        # new_kp = self.orientation_assignment()
        feats = self.get_local_descriptors(new_kp)

        return new_kp, feats

def read_from_csv(filename='keypoint.csv'):
    keypoint_from_csv = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            keypoint_from_csv.append(row)
    return keypoint_from_csv

if __name__=='__main__':
    im_dir = '../image/'
    feat_dir= '../csv/'
    csv_file = 'keypoint.csv'
    im_file = "building.jpeg"

    keypoint = read_from_csv(feat_dir+csv_file)
    image = Image.open(im_dir+im_file)
    kp = [[ int(kp[0]), int(kp[1]), float(kp[2])] for kp in keypoint]

    img_des = Harris_description(image, keypoint)
    new_kp, feats = img_des.get_features()