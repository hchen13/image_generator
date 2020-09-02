from datetime import datetime

import cv2
import numpy as np
from ISR.models import RRDN, RDN
from PIL import Image


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

if __name__ == '__main__':
    image_path = 'exports/c497/0.jpg'
    origin = cv2.imread(image_path)

    print('[info] initializing model...')
    tick = datetime.now()
    m1 = RRDN(weights='gans')
    m2 = RDN(weights='noise-cancel')
    tock = datetime.now()
    print(f"[info] complete, time elapsed: {(tock - tick).total_seconds():.1f}s.\n")

    print('[info] enlarging...')
    tick = datetime.now()
    enlarged = m1.predict(origin)
    tock = datetime.now()
    print(f"[info] done, time elapsed: {(tock - tick).total_seconds():.2f}s")

    print('[info] cancelling noise...')
    tick = datetime.now()
    cancelled = m2.predict(enlarged, by_patch_of_size=50)
    tock = datetime.now()
    print(f"[info] done, time elapsed: {(tock - tick).total_seconds():.2f}s\n")

    print(f"original image size: {origin.shape}, enlarged size: {enlarged.shape}, final size: {cancelled.shape}")
    show_image(origin, enlarged, cancelled, width=1000)