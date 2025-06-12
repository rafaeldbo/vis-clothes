from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

import torch
from torch import nn, Tensor
from torchvision import transforms

from .segmentation import cloth_segmentation_batch

COLOR_LABELS = [
    'Beige', 'Black', 'Blue', 'Brown', 
    'Green', 'Grey', 'Multicolor', 
    'Orange', 'Pink', 'Purple', 
    'Red', 'White', 'Yellow',
]
COLOR_IMG_SIZE = (64, 64)  # Tamanho da imagem de entrada para o modelo de cor

color_model = torch.jit.load("./models/jit_model_color_66.pt")

color_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Resize(COLOR_IMG_SIZE),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def apply_color_model(images: List[Tensor], device: str = "cpu") -> List[str]:

    color_model.to(device).eval()
    
    batch_tensor = torch.stack([color_transform(img) for img in images]).to(device)
    with torch.no_grad():
        color_results = color_model(batch_tensor).cpu().numpy()
        pred_colors = np.argmax(color_results, axis=1)
    
    return [COLOR_LABELS[color] for color in pred_colors]

CATEGORY_LABELS = [
    'Blouses', 'Dresses', 'Tops',
    'Trousers', 'Blazers', 'Knitwear',
    'Shirts', 'Coats', 'Jumpsuits',
    'Jackets', 'Skirts', 'Sweaters',
    'Vests', 'Cardigans', 'Shorts'
]

def apply_category_model(images: List[Tensor], device: str = "cpu") -> List[str]:
    # Placeholder for category model application
    return [np.random.choice(CATEGORY_LABELS) for _ in images]

DETAILS_LABELS = [
    'Pattern', 'Floral', 'Solid', 
    'Animal print', 'Checkers', 'Stripes'
]

def apply_detail_model(images: List[Tensor], device: str = "cpu") -> List[str]:
    # Placeholder for detail model application
    return [np.random.choice(DETAILS_LABELS) for _ in images]

def apply_cloth_models(images:List[np.ndarray], size:float=1, device:str="cpu") -> List[Dict[str, str]]:
    print(f"preprocessing images...")
    preprocessed_images = cloth_segmentation_batch(images, size, device)
    
    print(f"predicting...")
    pred_colors = apply_color_model(preprocessed_images, device)
    pred_categories = apply_category_model(preprocessed_images, device)
    pred_details = apply_detail_model(preprocessed_images, device)
    
    return [{'color': color, 'category': category, 'detail': detail} 
                for color, category, detail in zip(pred_colors, pred_categories, pred_details)]
    
    
