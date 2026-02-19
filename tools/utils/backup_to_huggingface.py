#!/usr/bin/env python3
"""
Script to backup checkpoint folders to HuggingFace Hub.
Each subfolder in the source directory will be uploaded to a separate HuggingFace repo.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List
import argparse


def get_subfolders(source_dir: str) -> List[str]:
    """Get all subfolders in the source directory."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise ValueError(f"Source directory {source_dir} does not exist")

    subfolders = [f.name for f in source_path.iterdir() if f.is_dir()]
    return sorted(subfolders)


def create_and_upload_repo(
    subfolder_name: str,
    source_dir: str,
    hf_username: str,
    hf_cli_path: str,
    repo_type: str = "model",
    private: bool = True,
    dry_run: bool = False
):
    """
    Create a HuggingFace repo and upload the subfolder contents.

    Args:
        subfolder_name: Name of the subfolder to upload
        source_dir: Source directory containing the subfolders
        hf_username: HuggingFace username
        hf_cli_path: Path to huggingface-cli executable
        repo_type: Type of repo ("model", "dataset", or "space")
        private: Whether to create a private repo
        dry_run: If True, only print what would be done without executing
    """
    repo_name = subfolder_name.lower().replace("_", "-")
    repo_id = f"{hf_username}/{repo_name}"
    subfolder_path = os.path.join(source_dir, subfolder_name)

    print(f"\n{'='*80}")
    print(f"Processing: {subfolder_name}")
    print(f"Repo ID: {repo_id}")
    print(f"Source path: {subfolder_path}")
    print(f"{'='*80}")

    if dry_run:
        print("[DRY RUN] Would create repo and upload files")
        return True

    try:
        # Create the repo (will skip if already exists)
        create_cmd = [
            hf_cli_path,
            "repo",
            "create",
            repo_id,
            "--type", repo_type,
        ]
        if private:
            create_cmd.append("--private")

        print(f"Creating repo: {' '.join(create_cmd)}")
        result = subprocess.run(create_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Repo created successfully: {repo_id}")
        else:
            # Repo might already exist
            if "already exists" in result.stderr.lower():
                print(f"ℹ Repo already exists: {repo_id}")
            else:
                print(f"⚠ Warning creating repo: {result.stderr}")

        # Upload the folder contents
        print(f"\nUploading files from {subfolder_path}...")
        upload_cmd = [
            hf_cli_path,
            "upload",
            repo_id,
            subfolder_path,
            ".",
            "--repo-type", repo_type,
        ]

        print(f"Upload command: {' '.join(upload_cmd)}")
        result = subprocess.run(upload_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Successfully uploaded to {repo_id}")
            return True
        else:
            print(f"✗ Error uploading: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Error processing {subfolder_name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Backup checkpoint folders to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be done
  python backup_to_huggingface.py --username YOUR_HF_USERNAME --dry-run

  # Upload all folders as private model repos
  python backup_to_huggingface.py --username YOUR_HF_USERNAME

  # Upload specific folders only
  python backup_to_huggingface.py --username YOUR_HF_USERNAME --folders GLC_EGO4D_Causal GLC_Egtea_Causal

  # Upload as public repos
  python backup_to_huggingface.py --username YOUR_HF_USERNAME --public

  # Upload as dataset repos instead of model repos
  python backup_to_huggingface.py --username YOUR_HF_USERNAME --repo-type dataset
        """
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default="/mnt/data1/jiali/tmp_output",
        help="Source directory containing checkpoint subfolders (default: /mnt/data1/jiali/tmp_output)"
    )

    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Your HuggingFace username"
    )

    parser.add_argument(
        "--repo-type",
        type=str,
        choices=["model", "dataset", "space"],
        default="model",
        help="Type of HuggingFace repo to create (default: model)"
    )

    parser.add_argument(
        "--public",
        action="store_true",
        help="Create public repos (default: private)"
    )

    parser.add_argument(
        "--folders",
        nargs="+",
        help="Specific folder names to upload (default: all folders)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually doing it"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folders that already have repos (requires checking each repo)"
    )

    args = parser.parse_args()

    # Find huggingface-cli in PATH
    hf_cli_path = shutil.which("huggingface-cli")
    if not hf_cli_path:
        print("Error: huggingface-cli not found in PATH. Please install it:")
        print("  pip install huggingface_hub")
        print("\nThen login with:")
        print("  huggingface-cli login")
        return

    print(f"Found huggingface-cli at: {hf_cli_path}")

    # Verify huggingface-cli is working
    try:
        result = subprocess.run(
            [hf_cli_path, "version"],
            capture_output=True,
            check=True
        )
        print(f"Using huggingface-cli version: {result.stdout.decode().strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running huggingface-cli: {e}")
        print("Please ensure huggingface_hub is properly installed:")
        print("  pip install huggingface_hub")
        return

    # Get folders to process
    if args.folders:
        folders = args.folders
        print(f"Will process {len(folders)} specified folders")
    else:
        folders = get_subfolders(args.source_dir)
        print(f"Found {len(folders)} folders in {args.source_dir}")

    if args.dry_run:
        print("\n" + "="*80)
        print("DRY RUN MODE - No actual operations will be performed")
        print("="*80)

    print(f"\nConfiguration:")
    print(f"  Source directory: {args.source_dir}")
    print(f"  HuggingFace username: {args.username}")
    print(f"  Repo type: {args.repo_type}")
    print(f"  Visibility: {'Public' if args.public else 'Private'}")
    print(f"  Folders to process: {len(folders)}")

    if not args.dry_run:
        response = input("\nProceed with upload? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Process each folder
    success_count = 0
    failed_folders = []

    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}]")
        success = create_and_upload_repo(
            subfolder_name=folder,
            source_dir=args.source_dir,
            hf_username=args.username,
            hf_cli_path=hf_cli_path,
            repo_type=args.repo_type,
            private=not args.public,
            dry_run=args.dry_run
        )

        if success:
            success_count += 1
        else:
            failed_folders.append(folder)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total folders: {len(folders)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_folders)}")

    if failed_folders:
        print("\nFailed folders:")
        for folder in failed_folders:
            print(f"  - {folder}")

    if not args.dry_run:
        print(f"\nYou can view your repos at: https://huggingface.co/{args.username}")


if __name__ == "__main__":
    main()
