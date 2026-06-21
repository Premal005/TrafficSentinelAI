import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent

def get_md5(file_path):
    h = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def main():
    target_md5 = "f6b442256e7194b15d1c89cc3513819c"
    print(f"Searching for files matching MD5={target_md5}...")
    
    # Check all files in project root and data/
    for p in PROJECT_ROOT.glob("**/*"):
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            file_md5 = get_md5(p)
            if file_md5 == target_md5:
                print(f"Match: {p.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
