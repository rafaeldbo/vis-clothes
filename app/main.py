from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import zipfile, io, os
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

PATH = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(PATH, "processed_images")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

@app.post("/zip")
async def upload_zip(zipFile: UploadFile = File(...)):
    images = []
    filenames = []

    with zipfile.ZipFile(io.BytesIO(await zipFile.read()), "r") as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith((".png", ".jpg", ".jpeg")):
                with zip_ref.open(file_name) as image_file:
                    image_data = np.frombuffer(image_file.read(), np.uint8)
                    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    images.append(image)
                    filenames.append(file_name)

    cloth_data, processed_images = apply_cloth_models(images)
    
    # Salva as imagens processadas para vizualização
    # for filename, image in zip(filenames, processed_images):
    #     filepath = os.path.join(SAVE_PATH, filename)
    #     if not cv2.imwrite(filepath, image):
    #         print(f"Error saving image {filename}")
    #         if os.path.exists(filepath):
    #             os.remove(filepath)
    
    return {"results": {filename: data for filename, data in zip(filenames, cloth_data)}}














