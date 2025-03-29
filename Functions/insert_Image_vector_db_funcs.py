import cv2
import chromadb
from deepface import DeepFace

# Custom Functions
from .image_hash_func import get_image_hash


def add_new_image_to_chroma_db(
    image_path: str,
    person_name: str,
    image_size: tuple = (160, 160),
    model_name: str = "GhostFaceNet",
    persist_directory: str = "chroma_db"
):
    """
    Detects, aligns, and embeds a single face from `image_path`, then adds the embedding
    to a persistent local Chroma database. The metadata "name" is set to `person_name`.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="face_embeddings")

    # Compute file hash as a unique ID
    try:
        file_hash = get_image_hash(image_path)
    except Exception as e:
        raise Exception(f"Error reading file {image_path}: {e}")

    # Check if this file hash already exists
    existing_ids = collection.get(ids=[file_hash])
    if existing_ids["ids"]:
        raise Exception(f"Image already exists in the collection (hash: {file_hash}).")

    # Detect and align faces using DeepFace
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="opencv",
            enforce_detection=True
        )
    except Exception as e:
        raise Exception(f"Face detection failed for {image_path}: {e}")

    if not faces:
        raise Exception(f"No face detected in {image_path}.")

    # Use the first detected face
    aligned_face = faces[0]["face"]
    aligned_face = cv2.resize(aligned_face, image_size)

    # Generate embedding
    try:
        representation = DeepFace.represent(
            aligned_face,
            model_name=model_name,
            enforce_detection=False
        )
    except Exception as e:
        raise Exception(f"Error generating embedding for {image_path}: {e}")

    if not representation or "embedding" not in representation[0]:
        raise Exception(f"Could not obtain a valid embedding for {image_path}.")

    embedding = representation[0]["embedding"]

    # Add to the Chroma collection (store the person's name in metadata)
    collection.add(
        ids=[file_hash],
        embeddings=[embedding],
        metadatas=[{"name": person_name}]
    )
    # Optionally, you could return a success message here.
    return f"Successfully added image for person '{person_name}'."