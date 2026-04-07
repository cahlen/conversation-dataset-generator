"""HuggingFace Hub upload functionality."""

import io
import logging
import time

logger = logging.getLogger(__name__)


def upload_to_hub(
    dataset_dict,
    repo_id: str,
    card_content: str | None = None,
    force: bool = False,
) -> bool:
    """Upload a DatasetDict to HuggingFace Hub.

    Args:
        dataset_dict: HuggingFace DatasetDict to upload.
        repo_id: Target repository ID (e.g., "username/dataset-name").
        card_content: Optional markdown content for README.md.
        force: Skip upload confirmation.

    Returns:
        True if upload succeeded, False otherwise.
    """
    from huggingface_hub import HfApi, HfFolder

    token = HfFolder.get_token()
    if not token:
        logger.error("HuggingFace token not found. Run: huggingface-cli login")
        return False

    if not force:
        try:
            confirm = input(f"Upload to {repo_id}? (yes/no): ")
            if confirm.lower() != "yes":
                logger.info("Upload cancelled.")
                return False
        except EOFError:
            logger.warning("No input available. Upload cancelled.")
            return False

    start = time.monotonic()
    try:
        dataset_dict.push_to_hub(repo_id, private=False)
        logger.info("Dataset pushed in %.2fs", time.monotonic() - start)
    except Exception as e:
        logger.error("Failed to push dataset: %s", e)
        return False

    if card_content:
        try:
            api = HfApi(token=token)
            readme_bytes = card_content.encode("utf-8")
            api.upload_file(
                path_or_fileobj=io.BytesIO(readme_bytes),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )
            logger.info("README.md uploaded.")
        except Exception as e:
            logger.warning("Failed to upload README.md: %s", e)

    return True
