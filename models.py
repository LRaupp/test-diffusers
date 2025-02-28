import os
import torch
import requests

from diffusers.models import AutoencoderKL
from diffusers import (
    StableDiffusionControlNetPipeline, ControlNetModel, 
    StableDiffusionPipeline, ModelMixin, FluxControlNetModel, 
    FluxPipeline, FluxControlNetPipeline, DiffusionPipeline,
    DPMSolverMultistepScheduler, ControlNetUnionModel,
    StableDiffusionXLControlNetUnionPipeline, StableDiffusionXLPipeline,
    DiffusionPipeline)

from typing import Literal, Tuple, Any, List
from compel import Compel
from packaging import version
from PIL import Image
from utils import plot_preprocessed_image

# https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit?gid=0#gid=0
# ADE20K Class -> Roads -> #8C8C8C
# ADE20K Class -> Buildings -> #B47878
# ADE20K Class -> Grass -> #04FA07
# ADE20K Class -> Water -> #3DE6FA
# ADE20K Class -> Sidewalk -> #EBFF07
# ADE20K Class -> Sky -> #06E6E6


# ControlNet
class ControlModes:
    openpose = "openpose"
    depth = "depth"
    hed = "hed"
    pidi = "pidi"
    scribble = "scribble"
    ted = "ted"
    canny = "canny"
    lineart = "lineart"
    anime_lineart = "anime_lineart"
    mlsd = "mlsd"
    normal = "normal"
    segment = "segment"
    tile = "tile"
    blur = "blur"
    gray= "gray"
    lq = "lq"


class CNHFModel:
    segment_model:str
    depth_model:str
    canny_model:str
    tile_model:str
    union_model:str
    union_input_order: Tuple[Tuple[str]]
    loader: ModelMixin

    @classmethod
    def is_union_model(cls):
        return hasattr(cls, "union_model") and getattr(cls, "union_model")

    @classmethod
    def load(cls, mode:Literal["segment", "depth", "edge", "tile", "union"], torch):
        if getattr(cls, f"{mode}_model"):
            if mode=="union":
                print(f"ControlNet images input order: {cls.union_input_order}")
            return cls.loader.from_pretrained(getattr(cls, f"{mode}_model"), torch_dtype=torch)
        else:
            available_models = [attr for attr in dir(cls) if attr.endswith("_model")]
            raise ValueError(f"Mode {mode} not available for model. Available models: {', '.join(available_models)}")


class ControlNetSD1_5Model(CNHFModel):
    segment_model = "lllyasviel/control_v11p_sd15_seg"
    depth_model = "lllyasviel/control_v11f1p_sd15_depth"
    canny_model = "lllyasviel/control_v11p_sd15_canny"
    tile_model = "lllyasviel/control_v11p_sd15_tile"
    loader = ControlNetModel


class ControlNetSDXL(CNHFModel):
    union_model = "xinsir/controlnet-union-sdxl-1.0"
    union_input_order = (
        (ControlModes.openpose,),
        (ControlModes.depth,),
        (ControlModes.hed, ControlModes.pidi, ControlModes.scribble, ControlModes.ted),
        (ControlModes.canny, ControlModes.lineart, ControlModes.anime_lineart, ControlModes.mlsd),
        (ControlModes.normal,),
        (ControlModes.segment,)
    )
    loader = ControlNetUnionModel


class FluxV1_DevControlNet(CNHFModel):
    union_model = "InstantX/FLUX.1-dev-Controlnet-Union"
    union_input_order = (
        (ControlModes.canny,),
        (ControlModes.tile,),
        (ControlModes.depth,),
        (ControlModes.blur,),
        (ControlModes.openpose,),
        (ControlModes.gray,),
        (ControlModes.lq,),
    )
    loader = FluxControlNetModel


# VAE
class VaeModel:
    model:str

    @classmethod
    def load(cls, torch_dtype):
        return AutoencoderKL.from_pretrained(cls.model, torch_dtype=torch_dtype)


class SDVaeFTMSE(VaeModel):
    model = "stabilityai/sd-vae-ft-mse"


class SDXLVaeFP16(VaeModel):
    model = "madebyollin/sdxl-vae-fp16-fix"


# StableDiffusion
class SDModel:
    controlnet_model: CNHFModel
    vae_model: VaeModel | None = None
    pipeline_loader: DiffusionPipeline
    cn_pipeline_loader: DiffusionPipeline
    pipe_extra_kwargs: dict | None = {}


class SDHFModel(SDModel):
    """Modelo carregado do HuggingFace"""
    model:str


class SD1_5Model(SDHFModel):
    model = "runwayml/stable-diffusion-v1-5"
    controlnet_model = ControlNetSD1_5Model
    vae_model = SDVaeFTMSE
    pipeline_loader = StableDiffusionPipeline
    cn_pipeline_loader = StableDiffusionControlNetPipeline


class SDXL1(SDHFModel):
    model = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model = ControlNetSDXL
    vae_model = SDXLVaeFP16
    pipeline_loader = StableDiffusionXLPipeline
    cn_pipeline_loader = StableDiffusionXLControlNetUnionPipeline
    pipe_extra_kwargs = {
        "variant": "fp16", 
        "use_safetensors": True
    }


class SDXL1Refiner(SDXL1):
    model = "stabilityai/stable-diffusion-xl-refiner-1.0"


class FluxV1_Dev(SDHFModel):
    model = "black-forest-labs/FLUX.1-dev"
    controlnet_model = FluxV1_DevControlNet
    pipeline_loader = FluxPipeline
    cn_pipeline_loader = FluxControlNetPipeline


class RealisticVisionV2(SD1_5Model):
    model = "SG161222/Realistic_Vision_V2.0"


class RealisticVisionV3(SD1_5Model):
    model = "SG161222/Realistic_Vision_V3.0_VAE"


class RealisticVisionV4(SD1_5Model):
    model = "SG161222/Realistic_Vision_V4.0_noVAE"


class RealisticVisionV5(SD1_5Model):
    model = "SG161222/Realistic_Vision_V5.0_noVAE"


class RealisticVisionV6(SD1_5Model):
    model = "SG161222/Realistic_Vision_V6.0_B1_noVAE"


#LoRa
class LocalLoraModel:
    model_file: str
    base_model: str
    reference_url: str
    download_url: str
    
    def __init__(self, base_local_path: str) -> None:
        self.base_local_path = base_local_path
    
    def _download_model(self):
        if not os.path.exists(self.full_model_path):
            print(f"Downloading model from {self.download_url} to {self.full_model_path}...")
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()
            
            os.makedirs(self.base_local_path, exist_ok=True)
            with open(self.full_model_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print("Download complete.")

    @property
    def full_model_path(self):
        return os.path.join(self.base_local_path, self.model_file)
    
    def __call__(self):
        self._download_model()
        return self.full_model_path


class ABirdsEyeViewOfArchitectureV1(LocalLoraModel):
    model_file = "A bird's-eye view of architecture.safetensors"
    base_model = SD1_5Model
    reference_url = "https://civitai.com/models/115392/a-birds-eye-view-of-architecture"


class ABirdsEyeViewOfArchitectureV3(LocalLoraModel):
    model_file = "A bird's-eye view of architecture3.0.safetensors"
    base_model = SD1_5Model
    reference_url = "https://civitai.com/models/121843/a-birds-eye-view-of-architecture30"


class AARGAerial(LocalLoraModel):
    model_file = "AARG_aerial-000018.safetensors"
    base_model = SD1_5Model
    reference_url = "https://civitai.com/models/87893/public-building-aerial-view"


class AerialViewV2(LocalLoraModel):
    model_file = "aerial view-V2.safetensors"
    base_model = SD1_5Model
    reference_url = "https://civitai.com/models/119054/aerial-view"


class FluxDStyleUrbanJungles(LocalLoraModel):
    model_file = "FLUXD-Style-Urban_Jungles-urjungle.safetensors"
    base_model = FluxV1_Dev
    reference_url = "https://civitai.com/models/1130475/urban-jungles"


class JZCG005RealisticCityPhotography(LocalLoraModel):
    model_file = "JZCG005-Realistic city photography 1.0.safetensors"
    base_model = SD1_5Model
    reference_url = "https://civitai.com/models/114604/jzcg005-realistic-city-photography"


class JZCGXL026(LocalLoraModel):
    model_file = "JZCGXL026.safetensors"
    base_model = SDXL1
    reference_url = "https://civitai.com/models/363965/jzcgxl026-aerial-view"


class UrbanRealisticCityBirdsEyeView(LocalLoraModel):
    model_file = "urbanrealistic_v1.safetensors"
    base_model = SD1_5Model
    reference_url = "https://civitai.com/models/104689/urbanrealistic"


class BirdsEyeViewUrbanDesignScenes(LocalLoraModel):
    model_file = "Bird's-eye view of urban design scenes.safetensors"
    base_model = SD1_5Model
    reference_url = "https://civitai.com/models/208499/birds-eye-view-of-urban-design-scenes"


# Enumeracao de otimizadores
class Optimizers:
    DEVICE_AWARE = 0
    UNET_TORCH_COMPILER = 1 # When using torch >= 2.0, you can improve the inference speed by 20-30% with torch.compile
    CPU_OFFLOAD = 2 # Pouca VRAM Não tá funcionando no colab
    VAE_SLICING = 3
    XFORMERS_MEMORY_ATTENTION = 4


class PlaceDiffusionModel:
    def __init__(self, 
                 base_diffusion_model:SDHFModel=None, 
                 pipeline_extra_kwargs:dict={},
                 controlnet_images:Tuple[Tuple[Any, str]]=None,
                 lora_model:LocalLoraModel=None,
                 use_dpm_scheduler:bool=True,
                 use_vae:bool=True,
                 pipe_optimizers:List[int]=[
                     Optimizers.DEVICE_AWARE, 
                     Optimizers.UNET_TORCH_COMPILER, 
                     #Optimizers.CPU_OFFLOAD, 
                     Optimizers.VAE_SLICING, 
                     Optimizers.XFORMERS_MEMORY_ATTENTION
                 ],
        ):
        """
        :param base_diffusion_model: Modelo diffuser a ser utilizado. Se lora_model for definido, este modelo é ignorado.
        :param controlnet_images: Imagens utilizadas de referência para o ControlNet. Deve seguir a seguinte organização: ((imagem, "mode"),)
            Onde: imagem-> Imagem a ser carregada | mode-> ControlNet mode ('canny', 'depth', 'segment')
        :param lora_model: LoRa model utilizado junto ao diffuser. Nesse caso, o base_diffusion_model é ignorado e é utilizado o modelo definido dentro do Modelo de LoRa.

        """
        self.use_vae = use_vae
        self._vae = None

        self.pipe_optimizers = pipe_optimizers

        self._pipeline:DiffusionPipeline = None
        self.pipeline_extra_kwargs = pipeline_extra_kwargs

        self._input_base_diffusion_model = base_diffusion_model
        self._base_diffusion_model = None        

        self._controlnets = None
        self.controlnet_images = controlnet_images

        self.lora_model = lora_model

        self.use_dpm_scheduler = use_dpm_scheduler

    @property
    def base_diffusion_model(self):
        if self.lora_model:
            return self.lora_model.base_model
        return self._input_base_diffusion_model

    @property
    def vae(self):
        if self._vae is None:
            if self.base_diffusion_model.vae_model and self.use_vae:
                self._vae = self.base_diffusion_model.vae_model.load(torch_dtype=self.torch_dtype)
        
        return self._vae

    @property
    def torch_dtype(self):
        return torch.float16 if torch.cuda.is_available() else torch.float32

    @property
    def device_name(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @property
    def default_pipeline_kwargs(self):
        return {
            "vae": self.vae,
            "torch_dtype": self.torch_dtype,
            **self.base_diffusion_model.pipe_extra_kwargs,
            **self.pipeline_extra_kwargs
        }
    
    @property
    def control_modes(self):
        return [m for i, m in self.controlnet_images]

    @property
    def valid_control_ref_images(self):
        """Imagens presentes no input controlnet_images. Exclui 0 (usado para informar que o control dessa posição não foi inputado) da listagem."""
        return [i for i, m in self.controlnet_images if isinstance(i, (Image.Image, torch.Tensor))]

    @property
    def valid_control_modes_index(self):
        """
        Índice das imagens válidas passadas em controlnet_images. 
        Como se espera que a listagem passada segue a ordem solicitada pelo modelo, o índice é o mesmo indicado no modelo para o control_mode.
        """
        return [self.control_modes.index(m) for i, m in self.controlnet_images if isinstance(i, (Image.Image, torch.Tensor))]

    @property
    def control_ref_images(self):
        return [i for i, m in self.controlnet_images]
    
    def preview_processed_images(self):
        for i, m in self.controlnet_images:
            if isinstance(i, (Image.Image, torch.Tensor)):
                plot_preprocessed_image(image_tensor=i, legend=m)

    @property
    def controlnets(self) -> List[Any] | Any:
        if (self._controlnets is None) and self.controlnet_images:
            
            if self.base_diffusion_model.controlnet_model.is_union_model():
                self._controlnets = self.base_diffusion_model.controlnet_model.load(mode="union", torch=self.torch_dtype)
            
            else:
                self._controlnets = []
                for mode in self.control_modes:
                    self._controlnets.append(
                        self.base_diffusion_model.controlnet_model.load(mode=mode, torch=self.torch_dtype)
                    )

        return self._controlnets

    @property
    def pipeline(self) -> DiffusionPipeline:
        if self._pipeline is None:
            
            if self.controlnet_images:
                self._pipeline = self.base_diffusion_model\
                .cn_pipeline_loader.from_pretrained(
                    self.base_diffusion_model.model,
                    controlnet=self.controlnets,
                    **self.default_pipeline_kwargs
                )
            
            else:
                self._pipeline = self.base_diffusion_model\
                    .pipeline_loader.from_pretrained(
                        self.base_diffusion_model.model,
                        **self.default_pipeline_kwargs
                    )
                
            if self.lora_model:
                self._pipeline.load_lora_weights(self.lora_model.full_model_path)
                self._pipeline.fuse_lora()
            
            if self.use_dpm_scheduler:
                self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self._pipeline.scheduler.config)

            if Optimizers.UNET_TORCH_COMPILER in self.pipe_optimizers:
                self._pipeline.unet = torch.compile(self._pipeline.unet, mode="reduce-overhead", fullgraph=True)    
                
            if Optimizers.CPU_OFFLOAD in self.pipe_optimizers:
                self._pipeline.enable_model_cpu_offload()
            
            if Optimizers.DEVICE_AWARE in self.pipe_optimizers and Optimizers.CPU_OFFLOAD not in self.pipe_optimizers:
                self._pipeline.to(self.device_name)
            
            if Optimizers.VAE_SLICING in self.pipe_optimizers:
                self._pipeline.enable_vae_slicing()
            
            if Optimizers.XFORMERS_MEMORY_ATTENTION in self.pipe_optimizers:
                self._pipeline.enable_xformers_memory_efficient_attention()

        return self._pipeline

    @property
    def pipe_compel(self):
        return Compel(tokenizer=self.pipeline.tokenizer, text_encoder=self.pipeline.text_encoder)
