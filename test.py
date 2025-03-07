from diffusers import DiffusionPipeline
import torch

print(torch.cuda.is_available())
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to('cuda')

prompt = "B&W sketch of a cute dog sitting on the beach"
images = pipe(prompt).images
for i in range(len(images)):
    images[i].save(f'./image-dog2-{i}.png')
print("finished")