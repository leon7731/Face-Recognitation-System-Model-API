# FastAPI
from fastapi import APIRouter


### Face Recognition Router ###
router = APIRouter(tags=["Face_Recognition"])


## Face RecognitionEndpoints ##
from .post_face_recognition import *