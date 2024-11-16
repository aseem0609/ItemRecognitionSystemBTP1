import os
import json
import shutil
import pandas as pd

# Define paths
DATASET_ROOT = "KagglePreparedData"  # Adjust the path as needed
OUTPUT_ROOT = "PreparedYOLOData"

# Define unique class names (adjust as per your dataset)
UNIQUE_SKU_CLASSES = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5']

def convert_annotation(phase, in_dir, out_dir, unique_sku_classes):
    """
    Converts annotations from JSON to YOLOv5 format.
    """
    in_images_path = os.path.join(in_dir, phase)
    out_images_path = os.path.join(out_dir, "images", phase)
    out_labels_path = os.path.join(out_dir, "labels", phase)

    os.makedirs(out_images_path, exist_ok=True)
    os.makedirs(out_labels_path, exist_ok=True)

    # Load JSON annotations
    annotations_file = os.path.join(in_dir, f"instances_{phase}.json")
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Convert JSON to DataFrames
    imgs_df = pd.DataFrame(data['images'])
    anns_df = pd.DataFrame(data['annotations'])
    categories_df = pd.DataFrame(data['categories'])

    # Filter categories and images
    category_ids = categories_df['id']
    anns_df = anns_df[anns_df['category_id'].isin(category_ids)]
    img_ids = anns_df['image_id']
    imgs_df = imgs_df[imgs_df['id'].isin(img_ids)]

    # Process each image
    for _, img in imgs_df.iterrows():
        imgw, imgh = img['width'], img['height']
        dw, dh = 1.0 / imgw, 1.0 / imgh
        img_src_path = os.path.join(in_images_path, img['file_name'])
        if not os.path.exists(img_src_path):
            continue
        img_dst_path = os.path.join(out_images_path, img['file_name'])
        label_dst_path = os.path.join(out_labels_path, img['file_name'].replace('.jpg', '.txt'))

        # Process annotations for the image
        img_anns = anns_df[anns_df['image_id'] == img['id']]
        labels = []
        for _, ann in img_anns.iterrows():
            cls_id = unique_sku_classes.index(categories_df[categories_df['id'] == ann['category_id']]['name'].item())
            cx, cy = dw * ann['bbox'][0], dh * ann['bbox'][1]
            sw, sh = dw * ann['bbox'][2], dh * ann['bbox'][3]
            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {sw:.6f} {sh:.6f}")

        # Save labels to file
        with open(label_dst_path, 'w') as label_file:
            label_file.write("\n".join(labels))

        # Copy image to destination
        shutil.copyfile(img_src_path, img_dst_path)

def main():
    # Create output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Process train, val, and test sets
    for phase in ['train', 'val', 'test']:
        print(f"Processing {phase} data...")
        convert_annotation(phase, DATASET_ROOT, OUTPUT_ROOT, UNIQUE_SKU_CLASSES)
        print(f"{phase.capitalize()} data processed successfully!")

if __name__ == "__main__":
    main()
