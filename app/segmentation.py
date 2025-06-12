from typing import Tuple, Union, List

import torch
from torchvision import transforms

import cv2
import numpy as np
from functools import reduce

from cloths_segmentation.pre_trained_models import create_model

model = create_model("Unet_2020-10-30")

initial_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225],
    )  
])

def pad_image(
    image: np.ndarray, 
    min_size: Tuple[int, int]=(0, 0),
    factor: int=32, 
    border: int=cv2.BORDER_CONSTANT, 
    color: Tuple[int, int, int]=(0,0,0)
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Adiciona padding centralizado à imagem para que suas dimensões sejam múltiplas de um fator específico e 
    também atenda a um tamanho mínimo especificado.

    Args:
        image (np.ndarray): Imagem a ser preenchida
        factor (int): Fator pelo qual as dimensões da imagem devem ser múltiplas
        min_size (Tuple[int, int]): Tamanho mínimo que a imagem deve ter (altura, largura)
        border (int): Tipo de borda a ser usada para o preenchimento
        color (Tuple[int, int, int]): Cor do padding

    Returns:
        (np.ndarray, Tuple[int, int, int, int]): Contém a imagem com padding preenchido (tamanho múltiplo de `factor`) 
            e as dimensões do padding como (pad_top, pad_bottom, pad_left, pad_right)
    """
    h, w = image.shape[:2]
    new_h = max((min_size[0] + factor - 1) // factor * factor, (h + factor - 1) // factor * factor)
    new_w = max((min_size[1] + factor - 1) // factor * factor, (w + factor - 1) // factor * factor)

    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left

    # Adiciona padding centralizado ao redor da imagem
    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, border, value=color)

    return padded_image, (pad_top, pad_bottom, pad_left, pad_right)


def unpad_image(
    padded_image: np.ndarray, 
    pads: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Remove o padding centralizado da imagem, retornando apenas a região original.

    Args:
        padded_image (np.ndarray): Imagem com padding adicionado
        pads (Tuple[int, int, int, int]): Dimensões do padding a ser removido (pad_top, pad_bottom, pad_left, pad_right)

    Returns:
        np.ndarray: Imagem com padding removido e tamanho original
    """
    pad_top, pad_bottom, pad_left, pad_right = pads
    
    if pad_bottom == 0:
        padded_image = padded_image[pad_top:, :]
    else:
        padded_image = padded_image[pad_top:-pad_bottom, :]
    
    if pad_right == 0:
        padded_image = padded_image[:, pad_left:]
    else:
        padded_image = padded_image[:, pad_left:-pad_right]

    return padded_image


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Redimensiona a imagem para se aproximar do tamanho ideal, mantendo as proporções.

    Args:
        image (np.ndarray): Imagem a ser redimensionada.
        target_size (Tuple[int, int]): Tamanho ideal desejado (altura, largura).

    Returns:
        np.ndarray: Imagem redimensionada mantendo proporções.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_h / h, target_w / w)  # Mantém a proporção original
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    
    return cv2.copyMakeBorder(cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR), 
                            pad_top, pad_bottom, pad_left, pad_right, 
                            cv2.BORDER_CONSTANT, value=(255, 255, 255))


def cloth_segmentation_batch(
    images: List[np.ndarray], 
    size: Union[float]=1, 
    device: str="cpu",
) -> List[Union[np.ndarray, None]]:
    
    if (n := len(images)) == 0:
        return []

    # Redimensiona as imagens se o tamanho for especificado mantendo a proporção
    if size > 0 and size < 1:
        images = [cv2.resize(img, (int(img.shape[1] * size), int(img.shape[0] * size)), interpolation=cv2.INTER_LINEAR) for img in images]
    # descobre o tamanho da maior imagem (esse será o tamanho de entrada do modelo)
    in_size = reduce(lambda size, img: (max(size[0], img.shape[0]), max(size[1], img.shape[1])), images, (0, 0))
     
    preprocessed_images = [None]*n
    pads_list = [None]*n

    # Aplica o padding (exigência do modelo de segmentação) e a transformação inicial em cada imagem
    for i, img in enumerate(images):
        padded_image, pads_list[i] = pad_image(img, factor=32, min_size=in_size)
        preprocessed_images[i] = initial_transform(padded_image)

    # Converte a lista de tensores em um único tensor e envia para o dispositivo especificado
    model.to(device).eval()
    batch_tensor = torch.stack(preprocessed_images).to(device)
    
    # Passa o batch de imagens pelo modelo de segmentação
    with torch.no_grad():
        predictions = model(batch_tensor)[:, 0]

    heights, widths = [None]*n, [None]*n
    segmented_images = [None]*n
    
    for i, prediction in enumerate(predictions):
        # Pega a mascara segmentada e remove o padding
        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad_image(mask, pads_list[i])
        white_pixels = np.argwhere(mask != 0)

        if len(white_pixels) == 0:
            segmented_images[i] = images[i]
            print(f"Warning: No white pixels found in mask of image {i}, keeping original image.")
        else:
            y_min, x_min = white_pixels.min(axis=0)
            y_max, x_max = white_pixels.max(axis=0)
            
            # corta a imagem para remover o fundo desnecessário
            cropped_image = images[i][y_min:y_max, x_min:x_max] 
            cropped_mask = mask[y_min:y_max, x_min:x_max]
            
            # remove o fundo da imagem original usando a máscara segmentada
            segmented_images[i] = np.where(cropped_mask[:, :, None] == 0, 255, cropped_image)
        
        segmented_images[i] = segmented_images[i]
        heights[i], widths[i], _ = segmented_images[i].shape
    # Calcula o tamanho de saída para todas as imagens, baseado na maior imagem após o corte
    out_size = (max(heights), max(widths))
    
    # redimensiona todas as imagens segmentadas para o tamanho de saída mantendo a proporção
    return [resize_image(segmented_images[i], out_size) for i in range(n)]