"""
VisionGuard Lite - Real SAFER-Activities Download Utility
Allows authenticating and downloading the official gated SAFER-Activities dataset.

Usage:
  python3 download_real_safer.py --token <HF_TOKEN>
  # OR if HF_TOKEN is already set in the environment:
  python3 download_real_safer.py
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.downloader import SAFERDatasetManager

def main():
    parser = argparse.ArgumentParser(description="Download official gated SAFER-Activities dataset")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face read token (with access to g-b-r-l/SAFER-Activities)")
    parser.add_argument("--output_dir", type=str, default="/content/VisionGuardLite/datasets/raw/safer_real", help="Destination folder")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("Checking repository access...")
    access_ok, msg = SAFERDatasetManager.check_access(token)
    if not access_ok:
        print("\n❌ ACCESS DENIED / UNAUTHENTICATED:")
        print(msg)
        print("\nHow to get access:")
        print("1. Visit https://huggingface.co/datasets/g-b-r-l/SAFER-Activities")
        print("2. Click 'Agree and access repository' under CC BY-NC-SA 4.0 license.")
        print("3. Generate a read token at https://huggingface.co/settings/tokens")
        print("4. Re-run: python3 download_real_safer.py --token YOUR_TOKEN")
        sys.exit(1)

    print("\n✅ Access confirmed! Proceeding to download...")
    res = SAFERDatasetManager.download_real_dataset(output_dir=args.output_dir, token=token)
    print("\n🎉 Download complete! Files saved to:", res['output_dir'])

if __name__ == '__main__':
    main()
