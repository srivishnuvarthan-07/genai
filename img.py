from diffusers import StableDiffusionPipeline
import torch

# Load the model (requires GPU for good performance)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
).to("cuda")  # use "cpu" if no GPU, but it's very slow

# Text prompt
prompt = "A white Fortuner drifting in the desert at sunset, ultra realistic, cinematic"

# Generate image
image = pipe(prompt).images[0]

# Show or save the image
image.show()
image.save("generated_image.png")
