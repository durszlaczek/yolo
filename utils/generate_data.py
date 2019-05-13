import cv2
import re
import os
import urllib

import pandas as pd
import numpy as np

from random import randint, random

MAX_SUBFRAME_WIDTH = 240
MIN_SUBFRAME_WIDTH = 120
MAX_SUBFRAME_HEIGHT = 120
MIN_SUBFRAME_HEIGHT = 70


def combine(img1, img2):
    if random() > 0.5:
        h2, w2 = img2.shape[:2]
        rand = random()
        img2 = cv2.resize(img2, (int(rand * w2), int(rand * h2)))
    h2, w2 = img2.shape[:2]
    h1 = int(random() * h2 / 2)
    w1 = int(random() * w2 / 2)

    img1 = cv2.resize(img1, (w1, h1))
    xmin = randint(0, w2 - w1)
    xmax = xmin + w1
    ymin = randint(0, h2 - h1)
    ymax = ymin + h1

    img2[ymin:ymax, xmin:xmax] = img1

    return img2, {
        'width': w2,
        'height': h2,
        'class': 'rect',
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax
    }


def generate_images(img_path, img_outfile, labels_outfile, desired_n=1000):
    img_urls = list(pd.read_csv(img_path)['OriginalURL'])
    n_images = len(img_urls)

    labels_df = pd.DataFrame(columns=['filename', 'width',
                                      'height', 'class',
                                      'xmin', 'ymin',
                                      'xmax', 'ymax'])
    labels_outfile = os.path.join(labels_outfile, 'labels.csv')

    for i in range(608, desired_n):
        print(i)
        try:
            n1, n2 = randint(0, n_images - 1), randint(0, n_images - 1)
            if n1 != n2:
                img1 = download_image(img_urls[n1])
                img2 = download_image(img_urls[n2])

                img, specs = combine(img1, img2)

                filename = 'rect-{}.jpg'.format(i)
                specs['filename'] = filename
                cv2.imwrite(os.path.join(img_outfile, filename), img)

                labels_df = labels_df.append(specs, ignore_index=True)
        except:
            print('failed.')
            labels_df.to_csv(labels_outfile, index=False)

    labels_df.to_csv(labels_outfile, index=False)


def download_image(url):
    request = urllib.request.urlopen(url)
    img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


if __name__ == '__main__':
    generate_images(img_path='/Users/aga/Documents/Projects/keras-yolo3/images_ids/image_ids.csv',
                    img_outfile='/Users/aga/Documents/Projects/keras-yolo3/train_image_folder',
                    labels_outfile='/Users/aga/Documents/Projects/keras-yolo3/train_annot_folder')
