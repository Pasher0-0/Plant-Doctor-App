from pathlib import Path
import argparse
import sys

try:
	from rembg import remove
	HAS_REMBG = True
except BaseException as e:
	# rembg may call sys.exit() if a backend like onnxruntime is missing,
	# which raises SystemExit (a BaseException). Catch BaseException to
	# avoid terminating the script; fall back to OpenCV method.
	print(f"rembg unavailable: {e}. Falling back to OpenCV grabCut.")
	HAS_REMBG = False

from PIL import Image
import io
import numpy as np

def remove_bg_rembg(img_bytes: bytes) -> Image.Image:
	out_bytes = remove(img_bytes)
	return Image.open(io.BytesIO(out_bytes)).convert("RGBA")

def remove_bg_grabcut(img: Image.Image) -> Image.Image:
	import cv2

	cv_img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
	mask = np.zeros(cv_img.shape[:2], np.uint8)
	h, w = mask.shape
	rect = (1, 1, max(1, w-2), max(1, h-2))
	bgdModel = np.zeros((1, 65), np.float64)
	fgdModel = np.zeros((1, 65), np.float64)
	cv2.grabCut(cv_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
	img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
	rgba = np.dstack((img_rgb, mask2 * 255))
	return Image.fromarray(rgba)

def process_file(in_path: Path, out_path: Path):
	try:
		img = Image.open(in_path)
	except Exception as e:
		print(f"Skipping {in_path}: cannot open ({e})")
		return

	try:
		if HAS_REMBG:
			with open(in_path, 'rb') as f:
				data = f.read()
			out_img = remove_bg_rembg(data)
		else:
			out_img = remove_bg_grabcut(img)
	except Exception as e:
		print(f"Primary method failed for {in_path}: {e}. Trying fallback.")
		try:
			out_img = remove_bg_grabcut(img)
		except Exception as e2:
			print(f"Fallback also failed for {in_path}: {e2}")
			return

	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_img.save(out_path, format='PNG')
	print(f"Saved {out_path}")

def main():
	parser = argparse.ArgumentParser(description="Batch remove backgrounds from images.")
	parser.add_argument('--input', '-i', type=str, default='Nitrogen(N)', help='Input folder')
	parser.add_argument('--output', '-o', type=str, default=None, help='Output folder (default: input_no_bg)')
	args = parser.parse_args()

	input_dir = Path(args.input)
	if not input_dir.exists() or not input_dir.is_dir():
		print(f"Input folder not found: {input_dir}")
		sys.exit(1)

	output_dir = Path(args.output) if args.output else input_dir.parent / (input_dir.name + '_no_bg')
	output_dir.mkdir(parents=True, exist_ok=True)

	exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
	for p in sorted(input_dir.iterdir()):
		if p.is_file() and p.suffix.lower() in exts:
			out_file = output_dir / (p.stem + '_no_bg.png')
			process_file(p, out_file)

if __name__ == '__main__':
	main()

# python background_remove.py --input "Nitrogen(N)"