import os
import random
import shutil
from PIL import Image, ImageDraw

def create_shape(shape, size, color):
    image = Image.new('RGB', (224, 224), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    x = random.randint(10, 224 - size)
    y = random.randint(10, 224 - size)

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

    return image, [x, y, x + size, y + size]

def generate_dataset(num_images, output_dir, shapes):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    for i in range(num_images):
        shape = random.choice(shapes)
        size = random.randint(50, 150)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        image, bbox = create_shape(shape, size, color)

        image_path = os.path.join(output_dir, 'images', f'{i:05d}.png')
        label_path = os.path.join(output_dir, 'labels', f'{i:05d}.txt')

        image.save(image_path)

        x_center = (bbox[0] + bbox[2]) / 2 / 224
        y_center = (bbox[1] + bbox[3]) / 2 / 224
        width = (bbox[2] - bbox[0]) / 224
        height = (bbox[3] - bbox[1]) / 224

        with open(label_path, 'w') as f:
            f.write(f"{shapes.index(shape)} {x_center} {y_center} {width} {height}")

def extend_dataset(output_dir, new_shapes, num_images):
    current_images = len(os.listdir(os.path.join(output_dir, 'images')))
    all_shapes = ['circle', 'square'] + new_shapes

    for i in range(num_images):
        shape = random.choice(new_shapes)
        size = random.randint(50, 150)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        image, bbox = create_shape(shape, size, color)

        image_path = os.path.join(output_dir, 'images', f'{current_images + i:05d}.png')
        label_path = os.path.join(output_dir, 'labels', f'{current_images + i:05d}.txt')

        image.save(image_path)

        x_center = (bbox[0] + bbox[2]) / 2 / 224
        y_center = (bbox[1] + bbox[3]) / 2 / 224
        width = (bbox[2] - bbox[0]) / 224
        height = (bbox[3] - bbox[1]) / 224

        with open(label_path, 'w') as f:
            f.write(f"{all_shapes.index(shape)} {x_center} {y_center} {width} {height}")

generate_dataset(1000, 'datasets/synthetic_dataset_v1', ['circle', 'square'])

shutil.copytree('datasets/synthetic_dataset_v1', 'datasets/synthetic_dataset_v2')
extend_dataset('datasets/synthetic_dataset_v2', ['triangle', 'pentagon'], 500)
