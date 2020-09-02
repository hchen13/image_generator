import os
from datetime import datetime
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from ISR.models import RRDN, RDN
from PIL import Image
from tqdm import tqdm


def show_image(*images, width=None, col=None, wait=0, title=None, destroy=False):
    import imutils

    if title is None:
        title = 'image'
    if len(images) == 1:
        image = images[0]
        if width is not None:
            image = imutils.resize(image, width=width)
        cv2.imshow(title, image)
        key = cv2.waitKey(wait) & 0xff
        if destroy:
            cv2.destroyAllWindows()
        return

    if width is None:
        width = 800
    if col is None:
        col = len(images)
    row = np.math.ceil(len(images) / col)
    _width = int(width / col)

    montages = imutils.build_montages(images, (_width, _width), (col, row))
    for montage in montages:
        cv2.imshow(title, montage)
        cv2.waitKey(wait)
        if destroy:
            cv2.destroyAllWindows()


def enhance(images):
    print('[info] initializing models...')
    tick = datetime.now()
    # m1 = RRDN(weights='gans')
    m2 = RDN(weights='noise-cancel')
    tock = datetime.now()
    print(f"[info] complete, time elapsed: {(tock - tick).total_seconds():.1f}s.\n")

    for image in tqdm(images):
        enhanced = m2.predict(image)
        # enhanced = m2.predict(enhanced, by_patch_of_size=500)
        yield enhanced


if __name__ == '__main__':
    print('[info] initializing models...')
    tick = datetime.now()
    # m1 = RRDN(weights='gans')
    m1 = RDN(weights='psnr-large')
    m2 = RDN(weights='noise-cancel')
    tock = datetime.now()
    print(f"[info] complete, time elapsed: {(tock - tick).total_seconds():.1f}s.\n")

    index = 497
    folder = f'c{index}'
    pattern = os.path.join('./exports', folder, '*.jpg')
    image_paths = glob(pattern)
    images = [cv2.imread(p) for p in image_paths]
    for i, origin in enumerate(images):
        e1 = m1.predict(origin)
        e2 = m2.predict(e1, by_patch_of_size=400)
        show_image(origin, e1, e2, col=2, width=1200)
        break
