import torch
from .pipeline_stable_diffusion_3_controlnet import StableDiffusion3CommonPipeline
import comfy.model_management as mm

class LoadSD3DiffusersPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
             "controlnet": (
            [   'InstantX/SD3-Controlnet-Canny',
                'InstantX/SD3-Controlnet-Canny_alpha_512',
            ],
            {
            "default": 'InstantX/SD3-Controlnet-Canny'
             }),
             "use_t5": ("BOOLEAN", {"default": False}),
             "hf_token": ("STRING", {"default": "",}),
            },
        }

    RETURN_TYPES = ("SD3PIPELINE",)
    RETURN_NAMES = ("sd3_pipeline",)
    FUNCTION = "loadmodel"
    CATEGORY = "DiffusersSD3"

    def loadmodel(self, controlnet, use_t5, hf_token):
        # load pipeline
        base_model = 'stabilityai/stable-diffusion-3-medium-diffusers'
        pipe = StableDiffusion3CommonPipeline.from_pretrained(
            base_model, 
            controlnet_list=[controlnet],
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
            "images": ("IMAGE", ),
            "prompt": ("STRING", {"multiline": True, "default": "",}),
            "n_prompt": ("STRING", {"multiline": True, "default": "",}),
            "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "control_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 20.0, "step": 0.01}),
            "controlnet_start": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.01}),
            "controlnet_end": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "process"
    CATEGORY = "DiffusersSD3"

    def process(self, sd3_pipeline, images, width, height, prompt, n_prompt, seed, steps, cfg, control_weight, controlnet_start, controlnet_end):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = sd3_pipeline
        pipe.to(device, torch.float16)
        B, H, W, C = images.shape

        #images = images.to(device)
        images = images.permute(0, 3, 1, 2)
        images = images * 2.0 - 1.0
        out = []
        for img in images:
            # controlnet config
            controlnet_conditioning = [
                dict(
                    control_index=0,
                    control_image=img.unsqueeze(0),
                    control_weight=control_weight,
                    control_pooled_projections='zeros'
                )
            ]

            generator = torch.Generator(device='cpu')
            generator.manual_seed(seed)
            controlnet_start_step = int(steps * controlnet_start)
            controlnet_end_step = int(steps * controlnet_end)
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
