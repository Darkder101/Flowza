#!/usr/bin/env python3
"""
Cleanup script for Flowza development environment
"""
import os  # noqa : F401
import shutil
import sys  # noqa : F401
from pathlib import Path


def cleanup_temporary_files():
    """Clean up temporary datasets and logs"""
    print("🧹 Cleaning up temporary files...")

    # Clean up temporary datasets
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        temp_files = [
            f for f in datasets_dir.glob("*_*.csv") if len(f.stem.split("_")[-1]) == 8  # noqa 
        ]
        for temp_file in temp_files:
            temp_file.unlink()
            print(f"Removed temporary dataset: {temp_file}")

    # Clean up logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log*"):
            log_file.unlink()
            print(f"Removed log file: {log_file}")

    # Clean up Python cache
    cache_dirs = list(Path(".").rglob("__pycache__"))
    for cache_dir in cache_dirs:
        shutil.rmtree(cache_dir)
        print(f"Removed cache directory: {cache_dir}")

    print("✅ Cleanup completed")


if __name__ == "__main__":
    cleanup_temporary_files()
