"""Place PNG images onto a white background and export them.

Usage:
    python white_backgruond.py [input_folder] [-o output_folder]

If no input folder is provided, the script will process the `N_no_bg` folder
in the current working directory and write outputs to `N_no_bg/white_bg`.
"""

from pathlib import Path
from PIL import Image
import argparse


def process_folder(input_dir: Path, output_dir: Path | None = None) -> None:
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "white_bg"
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = list(input_dir.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return

    for src in png_files:
        with Image.open(src).convert("RGBA") as im:
            white_bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            white_bg.paste(im, (0, 0), im)
            out_rgb = white_bg.convert("RGB")
            out_path = output_dir / src.name
            out_rgb.save(out_path, format="PNG")

    print(f"Processed {len(png_files)} images → {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Put PNGs on white background")
    parser.add_argument("input", nargs="?", default="N_no_bg", help="Input folder (default: N_no_bg)")
    parser.add_argument("-o", "--output", help="Output folder (default: input/white_bg)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Input folder does not exist: {input_path}")
        return

    output_path = Path(args.output) if args.output else None
    process_folder(input_path, output_path)


if __name__ == "__main__":
    main()
