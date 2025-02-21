import torch
import numpy as np
from PIL import Image

def preprocess_image(image: str, target_size: tuple = None):
    """
    Prepara uma imagem para processamento em um pipeline de difusão.
    
    :param image: Caminho do arquivo de imagem.
    :param target_size: Tamanho alvo (largura, altura) para redimensionamento. Padrão é None.
    :return: Tensor da imagem pré-processada no formato (batch, canais, altura, largura).
    """
    # Carregar a imagem
    image = Image.open(image)
    
    # Redimensionar se necessário
    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Converter para array NumPy
    image = np.array(image, dtype=np.float32) / 255.0  # Normalizar para [0,1]
    
    # Adicionar dimensão do canal se for escala de cinza
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)  # (H, W, 1)
    
    # Converter para formato (C, H, W) exigido pelo PyTorch
    image = np.transpose(image, (2, 0, 1))
    
    # Converter para tensor e adicionar dimensão do batch
    image_tensor = torch.tensor(image).unsqueeze(0)
    
    return image_tensor
