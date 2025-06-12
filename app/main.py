from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import zipfile, io
import numpy as np
import cv2

from .clothes_models import apply_cloth_models

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos HTTP
    allow_headers=["*"],  # Permite todos os cabeçalhos
)

@app.post("/zip")
async def upload_zip(zipFile: UploadFile = File(...)):
    images = []
    filenames = []

    # Ler o arquivo ZIP em memória
    with zipfile.ZipFile(io.BytesIO(await zipFile.read()), "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                with zip_ref.open(file_name) as image_file:
                    image_data = np.frombuffer(image_file.read(), np.uint8)
                    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    images.append(image)
                    filenames.append(file_name)

    classifications = apply_cloth_models(images, 0.5)
    return {"results": {filename: classification for filename, classification in zip(filenames, classifications)}}














