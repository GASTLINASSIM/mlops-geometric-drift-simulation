import os
import numpy as np
from PIL import Image, ImageDraw
import shutil

# --- CONFIGURATION ---
DATA_DIR = "data"
CLASSES = ["square", "circle"]
IMG_SIZE = 64

# ---------------------------------------------------------
# 👇 CHANGE THESE PARAMETERS FOR EXPERIMENTS
# N_SAMPLES = 50  # Number of images per class
# NOISE_LEVEL = 0.0  # 0.0 = Clean, 0.5 = Very Noisy

N_SAMPLES = 200  # Number of images per class (>= 500 for EXO1)
NOISE_LEVEL = 0.15  # 0.0 = Clean, 0.5 = Very Noisy (EXO1: > 0)
INVERT_COLORS = False  # EXO2 will use True
# ---------------------------------------------------------

def generate_dataset():
    # Clean previous data to avoid mixing versions
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    os.makedirs(DATA_DIR)
    print(f"🎨 Generating {N_SAMPLES * 2} images in '{DATA_DIR}' (Noise: {NOISE_LEVEL}).")

    for label in CLASSES:
        class_dir = os.path.join(DATA_DIR, label)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(N_SAMPLES):
            # Background (Black)
            img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
            draw = ImageDraw.Draw(img)

            # Random Shape Properties
            size = np.random.randint(10, 20)
            x = np.random.randint(10, 40)
            y = np.random.randint(10, 40)
            
            # Draw Shape
            if label == "square":
                draw.rectangle([x, y, x + size, y + size], fill=255)
            else:
                draw.ellipse([x, y, x + size, y + size], fill=255)

            # Add Noise (Salt & Pepper)
            if NOISE_LEVEL > 0:
                img_array = np.array(img)
                noise = np.random.random(img_array.shape)

                # Add white pixels randomly
                img_array[noise < NOISE_LEVEL] = 255
                img = Image.fromarray(img_array)

            # Save
            img.save(f"{class_dir}/{i}.png")

    print("✅ Generation complete!")

if __name__ == "__main__":
    generate_dataset()
