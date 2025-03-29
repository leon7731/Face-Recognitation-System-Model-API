
import cv2
import numpy as np
from deepface import DeepFace

def recognize_faces_in_image(
    image: np.ndarray,
    collection,
    model_name: str = "GhostFaceNet",
    image_size: tuple = (160, 160)
) -> np.ndarray:
    """
    Detects faces in the input image, extracts aligned faces using DeepFace,
    computes embeddings, queries the persistent Chroma collection for the closest match,
    and returns the annotated image with bounding boxes and recognized names.
    
    Args:
        image (numpy.ndarray): The input image in BGR format.
        collection: Chroma collection containing face embeddings.
        model_name (str): DeepFace model to use (default: "Facenet").
        image_size (tuple): Desired size for the aligned face image.
    
    Returns:
        numpy.ndarray: The annotated image with bounding boxes and recognized names.
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=image, 
            detector_backend="opencv", 
            enforce_detection=False
        )
    except Exception as e:
        print(f"Face detection failed: {e}")
        return image

    annotated_image = image.copy()

    # Adjust these constants to change bounding box thickness & text size
    BBOX_THICKNESS = 5
    TEXT_SCALE = 0.5
    TEXT_THICKNESS = 2

    for face in faces:
        facial_area = face["facial_area"]  # dict with x, y, w, h
        aligned_face = face["face"]
        aligned_face = cv2.resize(aligned_face, image_size)

        # Generate embedding
        try:
            representation = DeepFace.represent(
                aligned_face, 
                model_name=model_name, 
                enforce_detection=False
            )
        except Exception as e:
            print(f"Error generating embedding: {e}")
            continue

        if representation and len(representation) > 0:
            embedding = representation[0]["embedding"]
            # Query the collection for the nearest neighbor
            results = collection.query(query_embeddings=[embedding], n_results=1)
            
            # Retrieve the person's name from metadata (NOT the ID)
            if results["metadatas"] and results["metadatas"][0]:
                recognized_name = results["metadatas"][0][0].get("name", "Unknown")
            else:
                recognized_name = "Unknown"

            # Draw bounding box and label
            x = facial_area["x"]
            y = facial_area["y"]
            w = facial_area["w"]
            h = facial_area["h"]
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), BBOX_THICKNESS)
            cv2.putText(
                annotated_image, 
                recognized_name, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                TEXT_SCALE, 
                (0, 255, 0), 
                TEXT_THICKNESS
            )
    return annotated_image