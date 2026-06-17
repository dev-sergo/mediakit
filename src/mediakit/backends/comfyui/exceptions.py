from typing import Any


class ComfyUIError(Exception):
    """Base error for any ComfyUI client failure."""


class ComfyUIExecutionError(ComfyUIError):
    """ComfyUI reported execution_error for our prompt."""

    def __init__(self, *, prompt_id: str, payload: dict[str, Any]) -> None:
        self.prompt_id = prompt_id
        self.payload = payload
        node_type = payload.get("node_type")
        node_id = payload.get("node_id")
        message = payload.get("exception_message", "execution_error")
        super().__init__(
            f"ComfyUI execution failed for prompt {prompt_id} "
            f"(node={node_id}, type={node_type}): {message}"
        )


class ComfyUITimeoutError(ComfyUIError):
    """Waiting for a workflow exceeded the configured timeout."""
