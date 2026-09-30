"""Download the PublicHearingBR dataset from the Hugging Face Hub at a pinned revision."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from experiments.common.provenance import PROJECT_DIR

REPO_ID = "unicamp-dl/PublicHearingBR"
ALLOW_PATTERNS = ["*.jsonl", "*.md", "*.py"]
DEFAULT_REVISION = "2f84a44bc34df483e25c987f0ff86caad0ab3433"
DEFAULT_TARGET_DIR = PROJECT_DIR / "dataset"


def download_public_hearing_br(target_dir: Path, revision: str = DEFAULT_REVISION) -> Path:
    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=target_dir,
        allow_patterns=ALLOW_PATTERNS,
    )
    return Path(downloaded_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Download the {REPO_ID} dataset files from the Hugging Face Hub."
    )
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="dataset commit to download; the default is the commit every artifact was built from",
    )
    args = parser.parse_args()
    result_path = download_public_hearing_br(args.target_dir, args.revision)
    print(f"Dataset downloaded to {result_path}")


if __name__ == "__main__":
    main()
