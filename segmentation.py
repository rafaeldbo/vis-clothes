from typing import Tuple, Union

import torch
from torchvision import transforms

import cv2 as cv
import numpy as np

from cloths_segmentation.pre_trained_models import create_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def pad_image(image: np.ndarray, factor: int=32, border: int=cv.BORDER_CONSTANT) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Adiciona padding à imagem para que suas dimensões sejam múltiplas de um fator específico

    Args:
        image (np.ndarray): Imagem a ser preenchida
        factor (int): Fator pelo qual as dimensões da imagem devem ser múltiplas
        border (str): Tipo de borda a ser usada para o preenchimento

    Returns:
        (np.ndarray, tuple[int, int]): Contém a imagem com padding preenchido (tamanho múltiplo de `factor`) 
            e as dimensões do padding como (pad_h, pad_w)
    """
    h, w = image.shape[0], image.shape[1]
    new_h = (h + factor - 1) // factor * factor  # Garante múltiplo de factor
    new_w = (w + factor - 1) // factor * factor
    pad_h = new_h - h
    pad_w = new_w - w

    # Adiciona padding ao redor da imagem
    padded_image = cv.copyMakeBorder(image, 0, pad_h, 0, pad_w, border, value=(0, 0, 0))
    
    return padded_image, (pad_h, pad_w)


def unpad_image(padded_image: np.ndarray, pads: Tuple[int, int]) -> np.ndarray:
    """
    Remove o padding da imagem, retornando apenas a região original

    Args:
        padded_image (np.ndarray): Imagem com padding adicionado
        pads (tuple[int, int]): Dimensões do padding a ser removido


    Returns:
        np.ndarray: Imagem com padding removido e tamanho original
    """
    h, w = padded_image.shape[0], padded_image.shape[1]
    pad_h, pad_w = pads
    
    return padded_image[:h-pad_h, :w-pad_w]


model = create_model("Unet_2020-10-30")
model.eval()
model.to(device)

initial_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225],
    )  
])

def cloth_segmentation(image: np.ndarray, size: Union[float, None] = None) -> Union[np.ndarray, None]:
    """
    Realiza a segmentação de roupas em uma imagem usando um modelo pré-treinado 
    disponivel em https://github.com/ternaus/cloths_segmentation

    Args:
        image (np.ndarray): Imagem a ser segmentada
        size (float, optional): Fator de redimensionamento da imagem. 
            Se `None`, a imagem não será redimensionada
        
    Returns:
        (np.ndarray | None): Imagem segmentada com fundo branco 
            ou `None` se nenhuma roupa for detectada na imagem
    """
    
    img = image.copy()
    if size is not None:
        h,w = img.shape[0], img.shape[1]
        img = cv.resize(img, (int(w*size), int(h*size)), interpolation=cv.INTER_AREA)
        
    padded_image, pads = pad_image(img)
    tensor = initial_transform(padded_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(tensor)[0][0]
    
    if prediction is None:
        return None
    
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad_image(mask, pads)
    
    white_pixels = np.argwhere(mask == 1)

    if len(white_pixels) == 0:
        return None

    y_min, x_min = white_pixels.min(axis=0)
    y_max, x_max = white_pixels.max(axis=0)

    # Recortar a imagem original e a máscara
    cropped_image = img[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    # Criar uma imagem com fundo branco
    segmented_image = np.where(cropped_mask[:, :, None] == 0, 255, cropped_image)
    
    return segmented_image