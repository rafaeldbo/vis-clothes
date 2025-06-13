from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

import torch
from torch import nn, Tensor
from torchvision import transforms

from .segmentation import cloth_segmentation_batch

# Labels para cada classificação
## Cor
COLOR_LABELS = [
    'Beige', 'Black', 'Blue', 'Brown', 
    'Green', 'Grey', 'Multicolor', 
    'Orange', 'Pink', 'Purple', 
    'Red', 'White', 'Yellow',
]

## Categoria
CATEGORY_LABELS = [
    'Blouses', 'Dresses', 'Tops',
    'Trousers', 'Blazers', 'Knitwear',
    'Shirts', 'Coats', 'Jumpsuits',
    'Jackets', 'Skirts', 'Sweaters',
    'Vests', 'Cardigans', 'Shorts'
]

## Estampa
DETAILS_LABELS = [
    'Pattern', 'Floral', 'Solid', 
    'Animal print', 'Checkers', 'Stripes'
]

# Tamanho da imagem de entrada para o modelo de cor e estampa
IMG_SIZE = (64, 64) 

# Carregando os modelos
## Modelo para classificação por cor
color_model = torch.jit.load("./models/jit_model_color_66.pt")

## Modelo para classificação por estampa
details_model = torch.jit.load("./models/jit_model_detail_6.pt")

## Modelo para classificação por categoria
# category_model = torch.jit.load("./models/<nome_do_arquivo>.pt")

# Criando objetos Compose para os modelos
## Compose para modelo de classificação por cor
color_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Resize(IMG_SIZE),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

## Compose para modelo de classificação por estampa
details_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_SIZE),
])


def apply_color_model(images: List[Tensor], device: str = "cpu") -> List[str]:
    color_model.to(device).eval()
    
    batch_tensor = torch.stack([color_transform(img) for img in images]).to(device)
    with torch.no_grad():
        color_results = color_model(batch_tensor).cpu().numpy()
        pred_colors = np.argmax(color_results, axis=1)
    
    return [COLOR_LABELS[color] for color in pred_colors]


def apply_category_model(images: List[Tensor], device: str = "cpu") -> List[str]:
    # Placeholder for category model application
    return [np.random.choice(CATEGORY_LABELS) for _ in images]


def apply_detail_model(images: List[Tensor], device: str = "cpu") -> List[str]:
    details_model.to(device).eval()

    batch_tensor = torch.stack([details_transform(img) for img in images]).to(device)
    with torch.no_grad():
        details_results = details_model(batch_tensor).cpu().numpy()
        pred_details = np.argmax(details_results, axis=1)

    return [DETAILS_LABELS[detail] for detail in pred_details]


def apply_cloth_models(images:List[np.ndarray], size:int=512, device:str="cpu") -> Tuple[List[Dict[str, str]], List[np.ndarray]]:
    print(f"Preprocessing images...")
    preprocessed_images = cloth_segmentation_batch(images, size, device)
    
    print(f"Predicting...")
    pred_colors = apply_color_model(preprocessed_images, device)
    pred_categories = apply_category_model(preprocessed_images, device)
    pred_details = apply_detail_model(preprocessed_images, device)
    
    classifications = [{'color': color, 'category': category, 'detail': detail} 
                for color, category, detail in zip(pred_colors, pred_categories, pred_details)]
    
    return classifications, preprocessed_images
