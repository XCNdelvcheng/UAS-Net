"""
Build an index table mapping imaging data to corresponding segmentation files.
"""
import os
import glob
import pandas as pd

def main():
    # Configuration
    nifti_dir = "./data/raw_nifti"
    seg_dir = "./data/segmentation"
    output_csv = "./data/dataset_index.csv"

    rows = []

    for ct_path in glob.glob(os.path.join(nifti_dir, "*.nii*")):
        ct_filename = os.path.basename(ct_path)
        patient_id = ct_filename.split('_')[0]

        seg_files = glob.glob(os.path.join(seg_dir, f"*{patient_id}*.nii*"))
        seg_path = seg_files[0] if seg_files else None

        rows.append({
            "patient_id": patient_id,
            "ct_path": ct_path,
            "seg_path": seg_path
        })

    df = pd.DataFrame(rows)

    df['patient_id_num'] = df['patient_id'].astype(str).str.extract(r'(\d+)').astype(int)
    df = df.sort_values(by='patient_id_num').drop(columns='patient_id_num')

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Index table generated successfully: {output_csv}")

if __name__ == "__main__":
    main()
