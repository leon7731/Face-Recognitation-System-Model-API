
import os
import tempfile
from fastapi import  File, UploadFile, Form, HTTPException


# Custom Functions
from Functions.insert_Image_vector_db_funcs import add_new_image_to_chroma_db
from Functions.image_hash_func import get_image_hash

# token_router.py Router
from .face_upload_router import router


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    person_name: str = Form(...)
):
    # Create a temporary file to store the uploaded image.
    suffix = os.path.splitext(file.filename)[1]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_file_path = tmp.name
            # Write the file contents to the temporary file.
            tmp.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary file: {e}")

    try:
        # Call the custom function to process and add the image to the Chroma DB.
        message = add_new_image_to_chroma_db(temp_file_path, person_name, model_name="GhostFaceNet")
    except Exception as e:
        # Clean up the temp file before returning error.
        os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail=str(e))

    # Remove the temporary file after processing.
    os.remove(temp_file_path)
    return {"message": message}