"""
Convert DICOM data to NIfTI format using dcm2niix.
"""
import os
import subprocess
from tqdm import tqdm

def main():
    # Configuration
    dcm2niix_path = "dcm2niix"  # Assumes dcm2niix is added to system PATH
    input_root = "./data/raw_dicom"
    output_root = "./data/raw_nifti"
    compress = True

    os.makedirs(output_root, exist_ok=True)

    patient_folders = [f for f in os.listdir(input_root)
                       if os.path.isdir(os.path.join(input_root, f))]

    success_count = 0
    fail_list = []

    print(f"Found {len(patient_folders)} patient folders. Starting conversion...")
    for folder in tqdm(patient_folders, desc="Converting DICOM to NIfTI"):
        input_path = os.path.join(input_root, folder)
        output_name = folder
        
        cmd = [dcm2niix_path, "-o", output_root, "-f", output_name]
        if compress:
            cmd.extend(["-z", "y"])
        cmd.append(input_path)

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            success_count += 1
        except subprocess.CalledProcessError as e:
            fail_list.append(f"{folder}: {e.stderr}")
        except Exception as e:
            fail_list.append(f"{folder}: Unknown error - {str(e)}")

    print(f"\nConversion complete! Success: {success_count}, Failed: {len(fail_list)}")
    if fail_list:
        print("Failed list:")
        for err in fail_list:
            print(f" - {err}")
        with open(os.path.join(output_root, "conversion_error_log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(fail_list))

if __name__ == "__main__":
    main()
