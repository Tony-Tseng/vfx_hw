from stitch import Stitch
import numpy as np
import cv2

if __name__ == '__main__':
	s = Stitch('../data/data.txt', 900, 600)
	res = s.stitch()
	cv2.imwrite("../result_non_cropped.png", res)

	def get_first_non_empty(arr, t):
		for i, v in enumerate(arr):
			if v > t:
				return i
		return 0

	def get_last_non_empty(arr, t):
		for i, v in reversed(list(enumerate(arr))):
			if v > t:
				return i
		return len(arr)

	g = np.logical_and.reduce(res, axis=2)
	r = np.mean(g, axis=1)
	c = np.mean(g, axis=0)
	res = res[get_first_non_empty(r, 0.9):get_last_non_empty(r, 0.9), get_first_non_empty(c, 0.81):get_last_non_empty(c, 0.81)]
	cv2.imwrite('../result.png', res)