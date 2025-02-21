import os
import torch

from diffusers.models import AutoencoderKL
from diffusers import (
    StableDiffusionControlNetPipeline, ControlNetModel, 
    StableDiffusionPipeline, ModelMixin, FluxControlNetModel, 
    FluxPipeline, FluxControlNetPipeline, DiffusionPipeline,
    DPMSolverMultistepScheduler)

from typing import Literal, Tuple, Any, List



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
    loader = ControlNetModel


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


# StableDiffusion
class SDModel:
    controlnet_model: CNHFModel
    pipeline_loader: DiffusionPipeline
    cn_pipeline_loader: DiffusionPipeline


class SDHFModel(SDModel):
    model:str


class SD1_5Model(SDHFModel):
    model = "runwayml/stable-diffusion-v1-5"
    controlnet_model = ControlNetSD1_5Model
    pipeline_loader = StableDiffusionPipeline
    cn_pipeline_loader = StableDiffusionControlNetPipeline
    

class SDXL1(SDHFModel):
    model = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model = ControlNetSDXL
    pipeline_loader = StableDiffusionPipeline
    cn_pipeline_loader = StableDiffusionControlNetPipeline


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
    model_file:str 
    base_model:str
    reference_url:str
    
    def __init__(self, base_local_path:str) -> None:
        self.base_local_path = base_local_path
    
    @property
    def full_model_path(self):
        return os.path.join(self.base_local_path, self.model_file)


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


class PlaceDiffusionModel:
    def __init__(self, 
                 base_diffusion_model:SDHFModel=None, 
                 vae_model:str="stabilityai/sd-vae-ft-mse", 
                 pipeline_extra_kwargs:dict={},
                 controlnet_images:Tuple[Tuple[Any, str]]=None,
                 lora_model:LocalLoraModel=None,
                 use_dpm_scheduler:bool=True
        ):
        """
        :param base_diffusion_model: Modelo diffuser a ser utilizado. Se lora_model for definido, este modelo é ignorado.
        :param controlnet_images: Imagens utilizadas de referência para o ControlNet. Deve seguir a seguinte organização: ((imagem, "mode"),)
            Onde: imagem-> Imagem a ser carregada | mode-> ControlNet mode ('canny', 'depth', 'segment')
        :param lora_model: LoRa model utilizado junto ao diffuser. Nesse caso, o base_diffusion_model é ignorado e é utilizado o modelo definido dentro do Modelo de LoRa.

        """
        self._vae = None
        self.vae_model = vae_model

        self._pipeline = None
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
            self._vae = AutoencoderKL.from_pretrained(self.vae_model, torch_dtype=self.torch_dtype)
        
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
            **self.pipeline_extra_kwargs
        }
    
    @property
    def control_modes(self):
        return [m for i, m in self.controlnet_images]
    
    @property
    def control_ref_images(self):
        return [i for i, m in self.controlnet_images]
    
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
    def pipeline(self):
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

            # Melhora eficiência
            self._pipeline.to(self.device_name)
            self._pipeline.enable_vae_slicing()
            
            try:
                self._pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(e)

        return self._pipeline
