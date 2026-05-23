import os
import argparse
import cv2 as cv
import numpy as np


def reduce_colors(img, K=4, attempts=10):
	Z = img.reshape((-1, 3))
	Z = np.float32(Z)
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret, label, center = cv.kmeans(Z, K, None, criteria, attempts, cv.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	return res2


def process_folder(input_dir, output_dir, K=4, attempts=10, show=False):
	os.makedirs(output_dir, exist_ok=True)
	exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
	files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
	if not files:
		print('No images found in', input_dir)
		return
	for fname in files:
		in_path = os.path.join(input_dir, fname)
		img = cv.imread(in_path)
		if img is None:
			print('Warning: could not read', in_path)
			continue
		out = reduce_colors(img, K=K, attempts=attempts)
		out_path = os.path.join(output_dir, fname)
		cv.imwrite(out_path, out)
		print('Saved', out_path)
		if show:
			cv.imshow('reduced', out)
			key = cv.waitKey(500)  # show each image briefly
			if key == 27:  # ESC to stop showing
				cv.destroyAllWindows()
				show = False
	if show:
		cv.destroyAllWindows()


def main():
	parser = argparse.ArgumentParser(description='Batch color reduction using k-means')
	parser.add_argument('--input-dir', '-i', default=os.path.join(os.path.dirname(__file__), '..', 'Nitrogen(N)'), help='Input folder')
	parser.add_argument('--output-dir', '-o', default=os.path.join(os.path.dirname(__file__), '..', 'Nitrogen(N)_reduced'), help='Output folder')
	parser.add_argument('--k', '-k', type=int, default=4, help='Number of colors')
	parser.add_argument('--attempts', type=int, default=10, help='KMeans attempts')
	parser.add_argument('--show', action='store_true', help='Show images while processing')
	args = parser.parse_args()

	input_dir = os.path.abspath(args.input_dir)
	output_dir = os.path.abspath(args.output_dir)

	if not os.path.isdir(input_dir):
		print('Input directory does not exist:', input_dir)
		return

	process_folder(input_dir, output_dir, K=args.k, attempts=args.attempts, show=args.show)


if __name__ == '__main__':
	main()


# python color_reduce.py --input-dir "../Nitrogen(N)" --output-dir "../Nitrogen(N)_reduced" --k 4
