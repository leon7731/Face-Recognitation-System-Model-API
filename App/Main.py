from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

## Routers
from Routers.face_recognition import face_recognition_router
from Routers.face_upload import face_upload_router

### FastAPI App ### 
app = FastAPI()
  

### CORS Middleware ### 
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


### Default Route ###
@app.get("/")
def read_root():
    return {"message": "Face Recognition System API"} 



# # # Authentication Router
# # app.include_router(auth_router.router)


# Face Recognition Router
app.include_router(face_recognition_router.router)

# Face Upload Router
app.include_router(face_upload_router.router)




