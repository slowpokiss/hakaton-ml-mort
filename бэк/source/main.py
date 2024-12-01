from fastapi import FastAPI, UploadFile, File
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from utils import get_results

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/")
def hello():
    return {"message": "Hello, world!"}

@app.post("/upload-multiple-files/")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    uploaded_files = []
    for file in files:
        content = await file.read()
        
        with open(file.filename, "wb") as f:
            f.write(content)
        uploaded_files.append({"filename": file.filename, "content_type": file.content_type})

    return {"uploaded_files": uploaded_files}

@app.get("/get-result/")
async def get_result():
    metrics = get_results()

    return metrics
