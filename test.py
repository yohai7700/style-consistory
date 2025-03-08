from diffusers import DiffusionPipeline
import torch

print(torch.cuda.is_available())

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to('cuda')

prompt = "a watercolor painting of a dog in the forest"
generator = torch.Generator(device="cuda").manual_seed(100)  # Set your seed here
images = pipe(prompt, generator=generator).images
for i in range(len(images)):
    images[i].save(f'./image-dog2-forest-{i}.png')
print("finished")