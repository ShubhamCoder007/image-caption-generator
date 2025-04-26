#!/usr/bin/env python3
import os
import zipfile
import urllib.request
import argparse

GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP = "glove.6B.zip"
EMBED_DIR = "embeddings"

def download_glove(dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, GLOVE_ZIP)

    if not os.path.exists(zip_path):
        print(f"Downloading GloVe from {GLOVE_URL} …")
        urllib.request.urlretrieve(GLOVE_URL, zip_path)
        print("Download complete.")
    else:
        print("GloVe zip already present, skipping download.")

    # Unzip only the 300d file
    with zipfile.ZipFile(zip_path, "r") as z:
        target = "glove.6B.300d.txt"
        if not os.path.exists(os.path.join(dest_dir, target)):
            print(f"Extracting {target} …")
            z.extract(target, dest_dir)
            print("Extraction complete.")
        else:
            print(f"{target} already extracted, skipping.")

    # Optionally remove zip to save space
    # os.remove(zip_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download & unpack GloVe embeddings"
    )
    parser.add_argument(
        "--out_dir", default=EMBED_DIR,
        help="Directory to place glove.6B.zip and .txt"
    )
    args = parser.parse_args()
    download_glove(args.out_dir)
