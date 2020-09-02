import cv2
import numpy as np
from ISR.models import RRDN
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
    rdn = RRDN(weights='gans')
    print("[info] complete.\n")

    print('[info] predicting...')
    output = rdn.predict(origin)
    print("[info] done.\n")
    print(output.shape)
    show_image(origin, output, width=1000)