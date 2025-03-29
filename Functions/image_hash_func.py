import hashlib

def get_image_hash(image_path: str) -> str:
    """
    Compute MD5 hash for a file.
    """
    with open(image_path, "rb") as f:
        file_bytes = f.read()
    return hashlib.md5(file_bytes).hexdigest()