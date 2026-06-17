from enum import StrEnum


class JobStatus(StrEnum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"
