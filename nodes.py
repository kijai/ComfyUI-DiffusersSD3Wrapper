import torch
from .pipeline_stable_diffusion_3_controlnet import StableDiffusion3CommonPipeline
import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths
import os
import shutil

script_directory = os.path.dirname(os.path.abspath(__file__))

class LoadSD3DiffusersPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
             "controlnet_1": (
            [   'InstantX/SD3-Controlnet-Canny',
                'InstantX/SD3-Controlnet-Canny_alpha_512',
                'InstantX/SD3-Controlnet-Pose',
                'InstantX/SD3-Controlnet-Tile',
                'None'
            ],
            {
            "default": 'InstantX/SD3-Controlnet-Canny'
             }),
               "controlnet_2": (
            [   'InstantX/SD3-Controlnet-Canny',
                'InstantX/SD3-Controlnet-Canny_alpha_512',
                'InstantX/SD3-Controlnet-Pose',
                'None'
            ],
            {
            "default": 'None'
             }),
             "use_t5": ("BOOLEAN", {"default": False}),
             "hf_token": ("STRING", {"default": "",}),
            },
        }

    RETURN_TYPES = ("SD3PIPELINE",)
    RETURN_NAMES = ("sd3_pipeline",)
    FUNCTION = "loadmodel"
    CATEGORY = "DiffusersSD3"

    def loadmodel(self, controlnet_1, controlnet_2, use_t5, hf_token):
        # load pipeline
        controlnets_list = [controlnet_1]
        if controlnet_2 != 'None':
            controlnets_list.append(controlnet_2)
        controlnet_paths = []
        for controlnet in controlnets_list:
            model_name = controlnet.rsplit('/', 1)[-1]
            cn_model_path = os.path.join(folder_paths.models_dir, "diffusers", "SD3_controlnet", model_name)
            if not os.path.exists(cn_model_path):
                print(f"Downloading ControlNet model to: {cn_model_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=controlnet, 
                                    local_dir=cn_model_path, 
                                    local_dir_use_symlinks=False)
            if not os.path.exists(os.path.join(cn_model_path, "config.json")):
                source_path = os.path.join(script_directory, 'configs','config.json')
                destination_path = os.path.join(cn_model_path, 'config.json')
                shutil.copy(source_path, destination_path)
            controlnet_paths.append(cn_model_path)
            
        base_model = 'stabilityai/stable-diffusion-3-medium-diffusers'
        pipe = StableDiffusion3CommonPipeline.from_pretrained(
            base_model, 
            controlnet_list=controlnet_paths,
            token = hf_token
        )
        if not use_t5:
            pipe.text_encoder_3 = None
            pipe.tokenizer_3 = None
        
        #pipe.enable_model_cpu_offload()
           
        return (pipe,)
    
class SD3ControlNetSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "sd3_pipeline": ("SD3PIPELINE", ),
            "cn_images_1": ("IMAGE", ),
            "prompt": ("STRING", {"multiline": True, "default": "",}),
            "n_prompt": ("STRING", {"multiline": True, "default": "",}),
            "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "control_1_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 20.0, "step": 0.01}),
            "control_start": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.01}),
            "control_end": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
            "control_2_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 20.0, "step": 0.01}),
            "guess_mode": ("BOOLEAN", {"default": False}),

            },
            "optional": {
                "cn_images_2": ("IMAGE", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "process"
    CATEGORY = "DiffusersSD3"

    def process(self, sd3_pipeline, cn_images_1, width, height, prompt, n_prompt, seed, steps, cfg, control_1_weight, control_start, control_end, 
                control_2_weight, guess_mode, cn_images_2=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = sd3_pipeline
        pipe.to(device, torch.float16)
        B, H, W, C = cn_images_1.shape
        
        cn_images_1 = cn_images_1.permute(0, 3, 1, 2)
        cn_images_1 = cn_images_1 * 2.0 - 1.0

        out = []
        if B > 1:
            batch_pbar = ProgressBar(B)

        for img_1 in cn_images_1:
            controlnet_conditioning = [
                dict(
                    control_index=0,
                    control_image=img_1.unsqueeze(0),
                    control_weight=control_1_weight,
                    control_pooled_projections='zeros',
                    guess_mode = guess_mode
                )
            ]
            if cn_images_2 is not None:
                cn_images_2 = cn_images_2.permute(0, 3, 1, 2)
                cn_images_2 = cn_images_2 * 2.0 - 1.0
                for img_2 in cn_images_2:
                    controlnet_conditioning.append(
                        dict(
                            control_index=1,
                            control_image=img_2.unsqueeze(0),
                            control_weight=control_2_weight,
                            control_pooled_projections='zeros',
                            guess_mode = guess_mode
                        )
                    )

            generator = torch.Generator(device='cpu')
            generator.manual_seed(seed)
            controlnet_start_step = int(steps * control_start)
            controlnet_end_step = int(steps * control_end)
            print("cn start step: ",controlnet_start_step, "cn end step: ", controlnet_end_step)
            # infer
            results = pipe(
                prompt=prompt,
                negative_prompt=n_prompt,
                num_images_per_prompt=1,
                controlnet_conditioning=controlnet_conditioning,
                controlnet_start_step=controlnet_start_step,
                controlnet_end_step=controlnet_end_step,
                num_inference_steps=steps,
                guidance_scale=cfg,
                height=height,
                width=width,
                latents=None,
                generator=generator,
                output_type="pt",
            ).images[0]
            out.append(results)
            if B > 1:
                batch_pbar.update(1)
        tensor_out = torch.stack(out, dim=0)
        print(tensor_out.shape)
        tensor_out = tensor_out.permute(0, 2, 3, 1)
        tensor_out = tensor_out.cpu().float()
        

        return (tensor_out,)
    
NODE_CLASS_MAPPINGS = {
    "LoadSD3DiffusersPipeline": LoadSD3DiffusersPipeline,
    "SD3ControlNetSampler": SD3ControlNetSampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSD3DiffusersPipeline": "Load SD3DiffusersPipeline",
    "SD3ControlNetSampler": "SD3 ControlNet Sampler"
}
