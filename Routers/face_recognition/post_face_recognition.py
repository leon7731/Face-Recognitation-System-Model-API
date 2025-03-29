import io
import cv2
import numpy as np

import chromadb

from deepface import DeepFace

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

# Custom Functions
from Functions.face_recognition_func import recognize_faces_in_image

# token_router.py Router
from .face_recognition_router import router


@router.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    
    # Read the uploaded file bytes
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Load the persistent Chroma client and collection
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="face_embeddings")

    # Process the image using our face recognition function
    annotated = recognize_faces_in_image(image, collection, model_name="GhostFaceNet", image_size=(160, 160))
    
    # Encode the annotated image as JPEG
    success, encoded_image = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode image")

    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg") 