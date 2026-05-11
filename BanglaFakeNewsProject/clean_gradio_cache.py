"""Clean Gradio cache to fix 403 and file access errors"""
import os
import shutil
from pathlib import Path

# Get the base directory
BASE_DIR = Path(__file__).parent

# Directories to clean
cache_dirs = [
    BASE_DIR / "gradio_cached_examples",
    BASE_DIR / "flagged",
    Path.home() / ".gradio" / "cache"
]

print("🧹 Cleaning Gradio cache directories...")
for cache_dir in cache_dirs:
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print(f"✅ Cleaned: {cache_dir}")
        except Exception as e:
            print(f"⚠️ Could not clean {cache_dir}: {e}")
    else:
        print(f"ℹ️ Not found: {cache_dir}")

print("\n✅ Cache cleanup complete! Restart the app now.")
