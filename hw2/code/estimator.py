from feature_match import get_matches, find_good_homography
from harris import Harris
import cv2
import numpy as np
import numpy.linalg as LA

class Estimator:
	# def __init__(self):
	# 	self.surf = cv2.SIFT_create()

	def match(self, img1, img2):
		kpt1, desc1 = self.generate_features(img1)
		kpt2, desc2 = self.generate_features(img2)

		i1, i2 = get_matches(desc1, desc2, ratio=0.8)

		if len(i1) < 4:
			raise ValueError('Not enough matching pairs!')

		print(f'The number of matches: {len(i1)}')

		# cur = np.float32([kpt2[i].pt for i in i2])
		# prev = np.float32([kpt1[i].pt for i in i1])

		cur = np.float32([kpt2[i][:2] for i in i2])
		prev = np.float32([kpt1[i][:2] for i in i1])

		# import matplotlib.pyplot as plt

		# _, ax = plt.subplots(2)
		# ax[0].imshow(img1)
		# ax[0].scatter(kpt1[i1][:, 0], kpt1[i1][:, 1], c='r', s=3)
		# ax[0].axis('off')

		# ax[1].imshow(img2)
		# ax[1].scatter(kpt2[i2][:, 0], kpt2[i2][:, 1], c='r', s=3)
		# ax[1].axis('off')

		# plt.show()

		# H, s = cv2.estimateAffinePartial2D(cur, prev)
		# H = np.vstack((H, np.array([0, 0, 1])))
		H = find_good_homography(cur, prev)

		return H

	def generate_features(self, im):
		# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		# kpt, desc = self.surf.detectAndCompute(gray, None)

		h = Harris(im, diff_filter_name='origin')
		kpt, desc = h.get_keypoint()

		return kpt, desc
