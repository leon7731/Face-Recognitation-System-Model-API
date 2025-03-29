# FastAPI
from fastapi import APIRouter


### Face Upload Router ###
router = APIRouter(tags=["Face_Upload"])


## Face RecognitionEndpoints ##
from .post_new_face import *