import concurrent.futures
import glob
import os
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm


def process_image(img_path, input_dir, output_dir, low_threshold=50, high_threshold=100, type="canny"):
    try:
        stat_sum = 0.0
        stat_sq_sum = 0.0
        stat_count = 0

        if type == "HPF" or type == "canny" or type == "HPF_stack":
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image is None:
            tqdm.write(f"Warning: Could not read {img_path}. Skipping.")
            return f"Skipped: {img_path}", 0, 0, 0

        output = None

        if type == "canny":
            output = cv2.Canny(image, low_threshold, high_threshold)

        elif type == "HPF":
            # Single Channel HPF
            img_float = image.astype(np.float32)
            blurred = cv2.GaussianBlur(img_float, (17, 17), sigmaX=2, sigmaY=2)
            hpf = img_float - blurred + 127
            output = np.clip(hpf, 0, 255).astype(np.uint8)

            # Stats
            img_normalized = output / 255.0
            stat_sum = np.sum(img_normalized)
            stat_sq_sum = np.sum(img_normalized ** 2)
            stat_count = output.size

        elif type == "HPF_stack":
            # Multi-Channel HPF
            img_float = image.astype(np.float32)
            intensities = [1,2,4]
            hpfs = []


            for intensity in intensities:
                blurred = cv2.GaussianBlur(img_float, (17, 17), sigmaX=intensity, sigmaY=intensity)
                hpf = np.clip(img_float - blurred + 127, 0, 255).astype(np.uint8)
                hpfs.append(hpf)

            output = cv2.merge(hpfs)

            img_normalized = output / 255.0


            stat_sum = np.sum(img_normalized, axis=(0, 1))
            stat_sq_sum = np.sum(img_normalized ** 2, axis=(0, 1))
            stat_count = output.shape[0] * output.shape[1]

        elif type == "canny-overlap":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            output = image.copy()
            output[edges > 0] = [0, 0, 255]

        # --- Save the Result ---
        if output is not None:
            rel_path = os.path.relpath(img_path, input_dir)
            rel_dir = os.path.dirname(rel_path)
            current_output_dir = os.path.join(output_dir, rel_dir)
            os.makedirs(current_output_dir, exist_ok=True)

            basename = os.path.basename(img_path)
            name, _ = os.path.splitext(basename)
            output_name = f"{name}_2.png"
            output_path = os.path.join(current_output_dir, output_name)

            cv2.imwrite(output_path, output)
            return f"Processed: {img_path}", stat_sum, stat_sq_sum, stat_count

        return f"Skipped (No Op): {img_path}", 0, 0, 0

    except Exception as e:
        tqdm.write(f"Error processing {img_path}: {e}. Skipping.")
        return f"Error: {img_path}", 0, 0, 0


def batch_process_images(input_dir, output_dir, type="HPF", max_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    image_types = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_paths = []
    for ext in image_types:
        image_paths.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        return
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    print(f"Found {len(image_paths)} images. Processing with {max_workers} threads...")
    print(f"Mode: {type}")

    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixel_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(
            process_image,
            input_dir=input_dir,
            output_dir=output_dir,
            type=type,
        )
        results = list(tqdm(executor.map(process_func, image_paths), total=len(image_paths), desc="Processing"))

        for _, s, ss, c in results:
            total_sum += s
            total_sq_sum += ss
            total_pixel_count += c

    print(f"\nProcessing complete. Saved to '{output_dir}'.")

    if (type == "HPF" or type == "HPF_stack") and total_pixel_count > 0:

        global_mean = total_sum / total_pixel_count
        global_std = np.sqrt((total_sq_sum / total_pixel_count) - (global_mean ** 2))

        print("\n" + "=" * 40)
        print("STATISTICS")
        print("=" * 40)

        if isinstance(global_mean, np.ndarray):
            mean_str = "[" + ", ".join([f"{x:.4f}" for x in global_mean]) + "]"
            std_str = "[" + ", ".join([f"{x:.4f}" for x in global_std]) + "]"
        else:
            mean_str = f"[{global_mean:.4f}]"
            std_str = f"[{global_std:.4f}]"

        print(f"Mean: {mean_str}")
        print(f"Std:  {std_str}")
        print("-" * 40)
        print(f"Normalize(mean={mean_str}, std={std_str})")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    INPUT_FOLDER = "../data/Used_Cars_base"
    OUTPUT_FOLDER = "../data/Used_Cars_HPF_1ch_var"
    MAX_WORKERS = os.cpu_count()

    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created input folder '{INPUT_FOLDER}'.")
    else:
        batch_process_images(
            INPUT_FOLDER,
            OUTPUT_FOLDER,
            type="HPF",
            max_workers=MAX_WORKERS
        )