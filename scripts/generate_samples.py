import gc
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import load_image_backend

backend = load_image_backend()

print("Generating 5 photorealistic images...")
prompt = "portrait photograph of a college student in a coffee shop, arms crossed over chest, closed mouth subtle smile, warm natural lighting, shallow depth of field, 85mm f/1.8, professional photography, high detail"

negative_prompt = "open mouth, teeth visible, hands showing, fingers visible, distorted limbs, bad anatomy, deformed features, blurry, low quality, painting, drawing, sketch, cartoon, rendered, 3d, cgi, anime"

for i in range(5):
    print(f"Generating image {i+1}/5...")

    image = backend.generate(prompt, negative_prompt, steps=50, guidance=7.0, seed=i)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"images/realvis_{timestamp}.png"
    image.save(filename)
    print(f"✅ Saved {filename}")

    del image
    gc.collect()

print("Done! Check the images/ folder")
