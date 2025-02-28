import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def plot_preprocessed_image(image_tensor, legend:str=""):
    """
    Plots an image from a preprocessed tensor with a legend.
    
    :param image_tensor: Tensor of shape (batch, channels, height, width).
    :param legend: Texto da legenda a ser exibida.
    """
    image = image_tensor.squeeze(0).numpy()

    image = np.transpose(image, (1, 2, 0))

    image = np.clip(image, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")

    # Adicionar legenda na parte inferior da imagem
    plt.text(
        x=image.shape[1] / 2, y=image.shape[0] + 10, s=legend, 
        fontsize=12, ha="center", va="top", bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
    )

    plt.show()

def preprocess_image(image_path: str, target_size: tuple = None):
    """
    Prepara uma imagem para processamento em um pipeline de difusão.

    :param image_path: Caminho do arquivo de imagem.
    :param target_size: Tamanho alvo (largura, altura) para redimensionamento. Padrão é None.
    :return: Tensor da imagem pré-processada no formato (batch, canais, altura, largura).
    """
    # Carregar a imagem e converter para RGB (remove canal alfa)
    image = Image.open(image_path).convert("RGB")

    # Redimensionar se necessário
    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Converter para array NumPy e normalizar para [0,1]
    image = np.array(image, dtype=np.float32) / 255.0  

    # Converter para formato (C, H, W) exigido pelo PyTorch
    image = np.transpose(image, (2, 0, 1))

    # Converter para tensor e adicionar dimensão do batch
    image_tensor = torch.tensor(image).unsqueeze(0)

    return image_tensor

# Define the color palette (in RGB format)
PALETTE = {
    "Roads": (140, 140, 140),      # #8C8C8C
    "Buildings": (180, 120, 120),  # #B47878
    "Grass": (4, 250, 7),         # #04FA07
    "Water": (61, 230, 250),      # #3DE6FA
    "Sidewalk": (235, 255, 7),    # #EBFF07
    "Sky": (6, 230, 230)         # #06E6E6
}

# Convert palette to an array for fast computation
PALETTE_COLORS = np.array(list(PALETTE.values()))

def closest_color(pixel, palette=PALETTE_COLORS):
    """
    Finds the closest color in the given palette using Euclidean distance.
    
    :param pixel: The RGB pixel (R, G, B)
    :param palette: List of available colors in the palette
    :return: The closest color (R, G, B)
    """
    distances = np.linalg.norm(palette - pixel, axis=1)
    return tuple(palette[np.argmin(distances)])

def apply_palette(image_path, target_size=(256, 256)):
    """
    Loads an image, resizes it for faster processing, replaces its colors with the closest 
    ones from the predefined palette, and displays the modified image.
    
    :param image_path: Path to the input image.
    :param target_size: Tuple (width, height) for resizing (default: (256, 256)).
    :return: Processed PIL image with the new colors.
    """
    # Load image and convert to RGB
    image = Image.open(image_path).convert("RGB")

    # Resize the image (same as in preprocess_image)
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Convert image to NumPy array (without normalization)
    image_array = np.array(image, dtype=np.uint8)

    # Reshape for vectorized operations
    reshaped_pixels = image_array.reshape(-1, 3)

    # Find the closest color for each pixel
    new_pixels = np.array([closest_color(pixel) for pixel in reshaped_pixels])

    # Reshape back to original image dimensions
    new_image_array = new_pixels.reshape(target_size[1], target_size[0], 3)

    # Convert back to PIL image
    new_image = Image.fromarray(np.uint8(new_image_array))

    # Display the new image
    plt.figure(figsize=(6, 6))
    plt.imshow(new_image)
    plt.axis("off")
    plt.title("Image with Palette Colors")
    plt.show()

    return new_image