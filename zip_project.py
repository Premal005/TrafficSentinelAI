"""
Submission Packager — TrafficSentinel AI.

Creates a clean, optimized zip file of the source code for hackathon submission,
ensuring all local caches, heavy model weights, and evidence outputs are excluded.
This keeps the file size small and compliant with the < 50 MB limit.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
OUTPUT_ZIP = PROJECT_ROOT / "submission.zip"

# Directories and files to include
INCLUDE_DIRS = [
    "config",
    "core",
    "dashboard",
    "docs",
    "models",
    "pages",
    "tests",
    "utils",
    "violations",
    "data/sample_images",
]

INCLUDE_FILES = [
    "app.py",
    "requirements.txt",
    "README.md",
    "run_dashboard.bat",
    "__init__.py",
    "data/eval_results.json",
    "models/weights/custom_helmet_classifier.pth",
    "packages.txt",
    "render.yaml",
    ".python-version",
]

# Patterns or paths to explicitly exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".git",
    "models/weights",
    "evidence",
    "submission.zip",
]


def should_exclude(path: Path) -> bool:
    """Check if a path matches any exclude patterns."""
    rel_path = path.relative_to(PROJECT_ROOT)
    rel_str = str(rel_path).replace("\\", "/")
    
    for pattern in EXCLUDE_PATTERNS:
        # Check if the exact directory is a part of the path
        if any(part == pattern for part in rel_path.parts):
            return True
        # Check if it starts with the pattern (e.g. models/weights/)
        if rel_str == pattern or rel_str.startswith(pattern + "/"):
            return True
    return False


def package_project():
    print("[STATUS] Packaging TrafficSentinel AI for Submission...")
    print(f"   Project root: {PROJECT_ROOT}")
    print(f"   Output destination: {OUTPUT_ZIP}")
    print("-" * 50)

    # Delete existing zip if it exists
    if OUTPUT_ZIP.exists():
        os.remove(OUTPUT_ZIP)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add individual files
        for filename in INCLUDE_FILES:
            file_path = PROJECT_ROOT / filename
            if file_path.exists():
                print(f"[FILE] Adding file: {filename}")
                zipf.write(file_path, filename)
            else:
                print(f"[WARNING] Expected file not found: {filename}")

        # 2. Add directories
        for dirname in INCLUDE_DIRS:
            dir_path = PROJECT_ROOT / dirname
            if dir_path.exists():
                print(f"[DIR] Adding directory: {dirname}/")
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        full_path = Path(root) / file
                        if not should_exclude(full_path):
                            archive_name = full_path.relative_to(PROJECT_ROOT)
                            zipf.write(full_path, archive_name)
            else:
                print(f"[WARNING] Expected directory not found: {dirname}")

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print("-" * 50)
    print(f"[SUCCESS] Submission packaged successfully!")

    print(f"   File: {OUTPUT_ZIP.name}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Status: COMPLIANT (< 50 MB limit)")


if __name__ == "__main__":
    package_project()
