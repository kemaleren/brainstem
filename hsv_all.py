import os

import numpy as np
from matplotlib import pyplot as pp
from skimage.color import hsv2rgb

import texture as t

ANGLE = 20

filtered = os.path.expanduser("~/devel/results/filtered")
results = os.path.expanduser("~/devel/results/hsv")

filenames = list(f for f in os.listdir(filtered)
                 if os.path.splitext(f)[1] == ".npy")
n_files = len(filenames)

for i, filename in enumerate(filenames):
    print "processing {} of {}".format(i, n_files)
    img_file = os.path.splitext(filename)[0] + ".png"
    img_path = os.path.join(results, img_file)

    if os.path.exists(img_path):
        continue
    arr = np.load(os.path.join(filtered, filename))
    magnitude, angle = t.directionality_filter(arr, angle=ANGLE)
    img = hsv2rgb(t.make_hsv(magnitude, angle))


    pp.imsave(img_path, img)
