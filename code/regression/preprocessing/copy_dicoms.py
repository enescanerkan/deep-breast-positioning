# copy_dicoms.py
# This script:
# 1. Reads unique SOPInstanceUIDs from positioning_labels.csv (each image may have 2 rows: Pectoralis + Nipple)
# 2. Finds DICOM files from the VinDR dataset
# 3. Copies them into the folder structure expected by preprocessing:
#    quality / {StudyInstanceUID} / {SeriesInstanceUID} / {SOPInstanceUID}.dcm

import os
import shutil
import pandas as pd
from tqdm import tqdm
import argparse

def main(labels_csv, vindr_root, output_root):
    """
    Copy DICOM files from the VinDR dataset to the preprocessing folder structure.

    Args:
        labels_csv (str): Path to the positioning_labels.csv file.
        vindr_root (str): Root directory of VinDR DICOM images.
                          Expected structure: vindr_root/{StudyInstanceUID}/{SOPInstanceUID}.dicom
        output_root (str): Destination root directory for copied DICOMs.
                           Output structure: output_root/{StudyInstanceUID}/{SeriesInstanceUID}/{SOPInstanceUID}.dcm
    """
    print(f"Reading labels CSV: {labels_csv}")
    df = pd.read_csv(labels_csv)
    print(f"Total rows: {len(df)}")

    # Each SOPInstanceUID is a unique image
    # The CSV may contain multiple rows for the same image (Pectoralis, Nipple, etc.)
    unique_images = df.drop_duplicates(subset='SOPInstanceUID')[
        ['StudyInstanceUID', 'SOPInstanceUID', 'SeriesInstanceUID']
    ].reset_index(drop=True)

    print(f"Unique images: {len(unique_images)}")

    found = 0
    not_found = 0
    already_exists = 0
    not_found_list = []

    for _, row in tqdm(unique_images.iterrows(), total=len(unique_images), desc="Copying DICOMs"):
        study_uid = row['StudyInstanceUID']
        sop_uid   = row['SOPInstanceUID']
        series_uid = row['SeriesInstanceUID']

        # Source DICOM path (.dicom extension)
        src_path = os.path.join(vindr_root, study_uid, f"{sop_uid}.dicom")

        if not os.path.exists(src_path):
            not_found += 1
            not_found_list.append(src_path)
            continue

        # Destination path: output_root / StudyInstanceUID / SeriesInstanceUID / SOPInstanceUID.dcm
        dest_dir  = os.path.join(output_root, study_uid, series_uid)
        dest_path = os.path.join(dest_dir, f"{sop_uid}.dcm")

        if os.path.exists(dest_path):
            already_exists += 1
            continue

        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        found += 1

    # Summary
    print("\n" + "="*50)
    print(f"[OK] Copied           : {found}")
    print(f"[--] Already exists   : {already_exists}")
    print(f"[XX] Not found        : {not_found}")
    print("="*50)

    if not_found_list:
        log_path = os.path.join(output_root, "not_found.txt")
        os.makedirs(output_root, exist_ok=True)
        with open(log_path, 'w') as f:
            for p in not_found_list:
                f.write(p + "\n")
        print(f"\nNot found files logged to: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy DICOM files from VinDR dataset to preprocessing folder structure.")
    parser.add_argument('--labels_csv', type=str, default='../../labels/positioning_labels.csv',
                        help='Path to the positioning_labels.csv file.')
    parser.add_argument('--vindr_root', type=str, required=True,
                        help='Root directory of VinDR DICOM images.')
    parser.add_argument('--output_root', type=str, default='../quality',
                        help='Destination root directory for copied DICOMs.')
    args = parser.parse_args()

    main(args.labels_csv, args.vindr_root, args.output_root)
