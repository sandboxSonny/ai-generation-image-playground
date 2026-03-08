from diffusers import StableDiffusionXLPipeline
import torch
import gc

print("Loading RealVisXL V4.0 (photorealistic model)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

pipe = pipe.to("cuda")
pipe.vae.enable_slicing()

print("Generating 5 photorealistic images...")
prompt = "portrait photograph of a college student in a coffee shop, arms crossed over chest, closed mouth subtle smile, warm natural lighting, shallow depth of field, 85mm f/1.8, professional photography, high detail"

negative_prompt = "open mouth, teeth visible, hands showing, fingers visible, distorted limbs, bad anatomy, deformed features, blurry, low quality, painting, drawing, sketch, cartoon, rendered, 3d, cgi, anime"

for i in range(5):
    print(f"Generating image {i+1}/5...")
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=7.0,
        generator=torch.Generator("cuda").manual_seed(i)
    ).images[0]
    
    image.save(f"realvis_{i+1}.png")
    print(f"✅ Saved realvis_{i+1}.png")
    
    del image
    torch.cuda.empty_cache()
    gc.collect()

print("Done! Check realvis_1.png through realvis_5.png")
