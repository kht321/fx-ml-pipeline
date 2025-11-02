import os
from pathlib import Path
import argparse
import zipfile

def unzip_file(src_file_path: str, dest_dir_path: str) -> None:
    """
    Unzips a single .zip file from src_file_path into dest_dir_path.
    """
    print(f"Source file: {src_file_path}")
    print(f"Destination directory: {dest_dir_path}")
    src_file = Path(src_file_path)
    dest_dir = Path(dest_dir_path)
    if not src_file.is_file():
        raise ValueError(f"[error] Source file not found: {src_file}")
    if src_file.suffix.lower() != ".zip":
        raise ValueError(f"[error] Source is not a .zip file: {src_file}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(src_file, "r") as zf:
            zf.extractall(dest_dir)
        print(f"[unzip] Successfully extracted {src_file} -> {dest_dir}")
    
    except zipfile.BadZipFile as e:
        print(f"[error] Failed to unzip {src_file}. File corrupt or invalid.")
        raise e  
    
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-file", required=True)
    parser.add_argument("--bronze-path", required=True)
    args = parser.parse_args()

    unzip_file(args.raw_data_file, args.bronze_path)

if __name__ == "__main__":
    main()