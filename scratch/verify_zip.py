import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ZIP_PATH = PROJECT_ROOT / "submission.zip"

def main():
    if not ZIP_PATH.exists():
        print(f"Error: {ZIP_PATH} does not exist!")
        return

    print("Reading files in submission.zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zipf:
        file_list = zipf.namelist()
        
    print(f"Total files in zip: {len(file_list)}")
    
    target_files = [
        "models/download_models.py",
        "models/download_hf_models.py",
        "models/train_helmet_classifier.py",
        "core/evidence_generator.py",
        "models/weights/custom_helmet_classifier.pth",
    ]
    
    print("\n--- Checking Target Files ---")
    for f in target_files:
        present = f in file_list
        status = "OK" if present else "MISSING"
        print(f"  {f}: {status}")

if __name__ == "__main__":
    main()
