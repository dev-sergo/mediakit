import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from mediakit.config import settings


def save_upload(upload: UploadFile) -> Path:
    """Persist an uploaded file to storage/uploads and return its path."""
    settings.ensure_storage_dirs()
    max_bytes = settings.storage_max_upload_mb * 1024 * 1024
    # Read one extra byte to detect oversize without loading the entire file into memory.
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large — max {settings.storage_max_upload_mb} MB",
        )
    dest = settings.storage_uploads / f"{uuid.uuid4().hex}_{upload.filename or 'upload'}"
    dest.write_bytes(data)
    return dest
