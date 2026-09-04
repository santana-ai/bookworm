from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "unicamp-dl/PublicHearingBR"
ALLOW_PATTERNS = ["*.jsonl", "*.md", "*.py"]


def download_public_hearing_br(target_dir: Path) -> Path:
    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=ALLOW_PATTERNS,
    )
    return Path(downloaded_path)


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
    result_path = download_public_hearing_br(dataset_dir)
    print(f"Dataset downloaded to {result_path}")
