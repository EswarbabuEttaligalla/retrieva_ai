"""Server-Sent Events (SSE) serialization utilities for Retrieva AI.

Guarantees:
- Conforms strictly to the W3C SSE wire specification.
- Safely serializes event data (dicts, lists, primitives) into JSON.
- Handles multiline content correctly by prefixing each line with 'data: '.
- Terminates every frame with the required double newline ('\\n\\n').
"""

import json
from typing import Any


def format_sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event frame.

    Args:
        event: Event type name (e.g., 'status', 'token', 'citations', 'rejection', 'error', 'done').
        data: Payload object or string. Non-string objects are JSON-serialized.

    Returns:
        Formatted SSE text frame terminated with '\\n\\n'.
    """
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data)

    lines = payload.split("\n")
    data_payload = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event}\n{data_payload}\n\n"
