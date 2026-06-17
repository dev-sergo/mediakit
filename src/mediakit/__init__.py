"""mediakit — local AI media toolkit.

Three interfaces, one library:
  - Python package: import mediakit; await mediakit.compress(...)
  - CLI: mediakit compress input.jpg --format webp
  - HTTP server: POST /v1/ops/compress

Fast ops (native) execute in-process.
Slow ops (AI/ComfyUI) run via arq job queue with GPU concurrency=1.
"""

__version__ = "0.1.0"
