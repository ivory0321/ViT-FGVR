import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import math
import os
import glob
from tqdm import tqdm


MODEL_PATH = '../car-parts-seg/runs/segment/train-canny-50-100/weights/best.pt'
DATA_YAML_PATH = '../car-parts-seg/carparts-seg.yaml'
INPUT_DIRECTORY_PATH = '../data/Stanford_Cars_Edge'
OUTPUT_DIRECTORY_PATH = './output'

TILE_SIZE = 128
CANVAS_BG_COLOR = (255, 255, 255)  # White
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp']




def load_config(yaml_path):
    """Loads class names from the YAML file."""
    if not os.path.exists(yaml_path):
        tqdm.write(f"Error: YAML file not found at {yaml_path}")
        return None
    with open(yaml_path, 'r') as f:
        try:
            data = yaml.safe_load(f)
            if 'names' not in data:
                tqdm.write("Error: 'names' key not found in YAML.")
                return None
            return data['names']
        except yaml.YAMLError as exc:
            tqdm.write(f"Error parsing YAML file: {exc}")
            return None


def process_detections(result, orig_img):
    """
    Crops detected parts. If multiple parts of the same class are detected,
    it selects the one with the HIGHEST CONFIDENCE score.
    Returns a dictionary of {class_id: cropped_image}.
    """
    detected_parts_crops = {}

    if result.masks is None:
        return detected_parts_crops

    masks = result.masks.data
    boxes = result.boxes.data  # [x1, y1, x2, y2, conf, cls]
    orig_h, orig_w, _ = orig_img.shape

    # --- Step 1: Filter for highest confidence per class ---
    best_detections = {}  # Map: class_id -> (index_in_tensor, confidence_score)

    for i, box in enumerate(boxes):
        conf = float(box[4])  # Confidence is at index 4
        class_id = int(box[5])  # Class ID is at index 5 (or -1)

        if class_id not in best_detections:
            best_detections[class_id] = (i, conf)
        else:
            # If we already have this class, check if new one is better
            current_best_conf = best_detections[class_id][1]
            if conf > current_best_conf:
                best_detections[class_id] = (i, conf)

    # --- Step 2: Process only the best detection for each class ---
    for class_id, (i, conf) in best_detections.items():
        mask_raw = masks[i].cpu().numpy()
        mask_resized = cv2.resize(mask_raw, (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)

        # Create binary mask
        binary_mask_3d = np.stack([mask_resized] * 3, axis=-1) > 0.5
        masked_img = np.where(binary_mask_3d, orig_img, 0)

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, boxes[i, :4])

        # Crop
        cropped_part = masked_img[y1:y2, x1:x2]

        if cropped_part.size == 0:
            continue

        final_tile = create_square_tile(cropped_part)
        detected_parts_crops[class_id] = final_tile

    return detected_parts_crops


def create_square_tile(image):
    """
    Pads a non-square image to be square, then resizes it to TILE_SIZE.
    """
    h, w, _ = image.shape
    max_dim = max(h, w)
    square_canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    pad_x = (max_dim - w) // 2
    pad_y = (max_dim - h) // 2
    square_canvas[pad_y:pad_y + h, pad_x:pad_x + w] = image
    final_tile = cv2.resize(square_canvas, (TILE_SIZE, TILE_SIZE),
                            interpolation=cv2.INTER_AREA)
    return final_tile


def process_single_image(model, image_path, input_base_dir, output_base_dir):
    """
    Runs segmentation, calculates dynamic grid, and saves output.
    Returns: Number of segments found (int).
    """
    try:
        # --- 1. Load Image ---
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            tqdm.write(f"Error: Could not read image {image_path}, skipping.")
            return 0

        # --- 2. Run Prediction ---
        results = model.predict(orig_img, conf=0.25, verbose=False)
        result = results[0]

        # --- 3. Process Detections (Updated Logic) ---
        detected_parts_map = process_detections(result, orig_img)
        num_detected = len(detected_parts_map)

        if num_detected == 0:
            return 0

        # --- 4. Define Output Paths ---
        relative_path = os.path.relpath(image_path, input_base_dir)
        rel_dir, base_name = os.path.dirname(relative_path), os.path.basename(relative_path)
        file_name, _ = os.path.splitext(base_name)
        full_output_subdir = os.path.join(output_base_dir, rel_dir)
        os.makedirs(full_output_subdir, exist_ok=True)
        concat_output_path = os.path.join(full_output_subdir, f"{file_name}_concat.jpg")

        # --- 5. Dynamic Grid Calculation ---
        grid_dim = int(math.ceil(math.sqrt(num_detected)))

        canvas_size = grid_dim * TILE_SIZE
        canvas = np.full((canvas_size, canvas_size, 3),
                         CANVAS_BG_COLOR, dtype=np.uint8)

        i = 0
        for class_id, part_img in detected_parts_map.items():
            row = i // grid_dim
            col = i % grid_dim

            y_offset = row * TILE_SIZE
            x_offset = col * TILE_SIZE

            canvas[y_offset:y_offset + TILE_SIZE, x_offset:x_offset + TILE_SIZE] = part_img
            i += 1

        # --- 6. Save Final Concatenated Image ---
        cv2.imwrite(concat_output_path, canvas)

        return num_detected

    except Exception as e:
        tqdm.write(f"!! CRITICAL ERROR processing {image_path}: {e}")
        return 0


def main():
    print("Starting BATCH car part segmentation...")

    # --- 1. Load Configuration ---
    class_names = load_config(DATA_YAML_PATH)
    if class_names is None:
        return

    # --- 2. Load Model ---
    try:
        model = YOLO(MODEL_PATH)
        print(f"Successfully loaded model: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}. Check path. Error: {e}")
        return

    # --- 3. Prepare Output Directory ---
    os.makedirs(OUTPUT_DIRECTORY_PATH, exist_ok=True)

    # --- 4. Find All Images ---
    image_paths = []
    clean_input_dir = os.path.normpath(INPUT_DIRECTORY_PATH)

    for ext in IMAGE_EXTENSIONS:
        search_pattern = os.path.join(clean_input_dir, '**', ext)
        image_paths.extend(glob.glob(search_pattern, recursive=True))

    if not image_paths:
        print(f"Error: No images found in {clean_input_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")

    # --- 5. Loop and Process ---
    total_segments_global = 0

    for image_path in tqdm(image_paths, desc="Processing Images", unit="img"):
        count = process_single_image(model, image_path, clean_input_dir, OUTPUT_DIRECTORY_PATH)
        total_segments_global += count

    # --- 6. Show Final Count ---
    print("\n" + "=" * 40)
    print(f"PROCESSING COMPLETE")
    print(f"Total Images Processed: {len(image_paths)}")
    print(f"Total Segments Found:   {total_segments_global}")
    print("=" * 40)


if __name__ == "__main__":
    main()