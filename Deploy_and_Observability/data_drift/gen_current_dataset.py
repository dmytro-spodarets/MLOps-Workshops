import os
import random
import shutil
from PIL import Image, ImageDraw

def create_shape(shape, size, color):
    image = Image.new('RGB', (224, 224), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    x = random.randint(10, 214)
    y = random.randint(10, 214)

    if shape == 'circle':
        draw.ellipse([x, y, x + size, y + size], fill=color)
    elif shape == 'triangle':
        draw.polygon([(x, y + size), (x + size, y + size), (x + size // 2, y)], fill=color)
    elif shape == 'square':
        draw.rectangle([x, y, x + size, y + size], fill=color)
    elif shape == 'pentagon':
        points = [
            (x + size // 2, y),
            (x + size, y + size // 3),
            (x + 4 * size // 5, y + size),
            (x + size // 5, y + size),
            (x, y + size // 3)
        ]
        draw.polygon(points, fill=color)

    return image

def generate_dataset(folder, shapes, num_images_per_shape=50):
    os.makedirs(folder, exist_ok=True)
    for shape in shapes:
        for i in range(num_images_per_shape):
            size = random.randint(20, 100)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = create_shape(shape, size, color)
            img.save(os.path.join(folder, f'{shape}_{i}.png'))

generate_dataset('datasets/current_dataset_v1', ['circle', 'square'], num_images_per_shape=100)

shutil.copytree('datasets/current_dataset_v1', 'datasets/current_dataset_v2')
generate_dataset('datasets/current_dataset_v2', ['triangle', 'pentagon'], num_images_per_shape=100)
