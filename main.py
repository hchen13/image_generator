from pathlib import Path

import numpy as np
import torch
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample
from sklearn.externals._pilutil import toimage
from tqdm import tqdm

from enhance import enhance

print('[info] detecting device...')
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[info] device: {device}')
output_root = './exports/'


def load_model(resolution: int=512):
    print('[info] loading pre-trained model...')
    model_name = f'biggan-deep-{resolution}'
    pretrained_path = Path.home().joinpath('.pytorch_pretrained_biggan')
    model = BigGAN.from_pretrained(pretrained_path)
    print('[info] loading complete.\n')
    return model


def generate_images(num_images, class_index, model, truncation: float=1.):
    print("[info] initializing input variables...")
    batch_size = min(10, num_images)
    class_vec = one_hot_from_int(class_index, batch_size=num_images) * .97
    noise_vec = truncated_noise_sample(truncation=truncation, batch_size=num_images)

    noise_tensor = torch.tensor(noise_vec, dtype=torch.float)
    class_tensor = torch.tensor(class_vec, dtype=torch.float)

    print('[info] preparing model inference...')
    noise_tensor = noise_tensor.to(device)
    class_tensor = class_tensor.to(device)

    print(f'[info] generating {num_images} images of class index {class_index}...')
    images = []
    for i in range(0, num_images, batch_size):
        torch.cuda.empty_cache()
        noise_feed = noise_tensor[i : i + batch_size]
        class_feed = class_tensor[i : i + batch_size]
        with torch.no_grad():
            output = model(noise_feed, class_feed, truncation)
        output_cpu = output.cpu().data.numpy()
        for out in output_cpu:
            image = np.array(toimage(out))
            images.append(image)
    print('[info] done.\n')
    torch.cuda.empty_cache()
    return images


def save_images(images, folder):
    import os
    import cv2
    dir = os.path.join(output_root, folder)
    os.makedirs(dir, exist_ok=True)
    for i, image in enumerate(images):
        path = os.path.join(dir, f'{i}.jpg')
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image_bgr)


if __name__ == '__main__':
    class_indices = [497]
    number_images = 20


    model = load_model()
    model = model.to(device)

    for index in class_indices:
        images = generate_images(number_images, index, model, truncation=.9)
        save_images(images, f'c{index}')
        img_gen = enhance(images)
        save_images(img_gen, f'c{index}-enhanced')
