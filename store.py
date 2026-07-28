import uuid

_store: dict[str, bytes] = {}

def save_image(image_bytes: bytes) -> str:
    scan_id = str(uuid.uuid4())
    _store[scan_id] = image_bytes
    return scan_id

def get_image(scan_id: str) -> bytes | None:
    return _store.get(scan_id)

