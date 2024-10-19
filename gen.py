from PIL import Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained("hakurei/waifu-diffusion", torch_dtype=torch.float16)

# Load the img2img pipeline
#pipe = StableDiffusionImg2ImgPipeline.from_pretrained("hakurei/waifu-diffusion", torch_dtype=torch.float16)

# Set the sampling method to Euler A
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Disable the safety checker
def dummy_safety_checker(images, clip_input):
    return images, True

pipe.safety_checker = dummy_safety_checker

# Enable memory-efficient attention
pipe.enable_attention_slicing()

pipe.to("cuda")

# Load the reference image (the one you want to bias the result on)
init_image = Image.open("zReferences/ref.png").convert("RGB")
init_image = init_image.resize((512, 512))  # Resize to match model's input size

# Define your prompt
#prompt = "masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, watercolor, night, turtleneck"
prompt = "masterpiece, best quality, 1girl, green hair, looking at viewer, beanie, naked"

# Define settings for img2img
strength = 0.6  # The strength parameter controls how much the generated image will deviate from the original
guidance_scale = 6  # Higher guidance scale means the model will follow the prompt more closely

# Generate the image based on the input image and the prompt using Euler A sampler
image = pipe(init_image = init_image, prompt=prompt, strength=strength, guidance_scale=guidance_scale,num_inference_steps=25, height=512, width=512).images[0]

# Save the generated image
image.save("output.png")
