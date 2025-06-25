import argparse
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import sys

def download_dataset(dataset_ref, dest_dir):
    """
    Downloads and unzips a dataset from Kaggle.
    """
    try:
        print("Initializing Kaggle API...")
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API authenticated successfully.")

        print(f"Downloading dataset '{dataset_ref}' to '{dest_dir}'...")
        
        os.makedirs(dest_dir, exist_ok=True)

        api.dataset_download_files(dataset_ref, path=dest_dir, unzip=True)

        # Verify that files were actually downloaded
        downloaded_files = os.listdir(dest_dir)
        if not downloaded_files:
            print("Error: Download command finished, but no files were found. This often indicates an authentication issue or an invalid dataset reference.", file=sys.stderr)
            exit(1)

        print(f"Download and unzip completed successfully. Files found: {len(downloaded_files)}")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download a dataset from Kaggle.')
    parser.add_argument('--ref', type=str, required=True, help='Dataset reference (e.g., owner/dataset-slug)')
    parser.add_argument('--dest', type=str, required=True, help='Destination directory')
    
    args = parser.parse_args()
    
    download_dataset(args.ref, args.dest) 