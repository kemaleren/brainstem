import os

from matplotlib import pyplot as pp
from skimage.color import hsv2rgb

import texture as t
import brainstem as b

RESULTS = os.path.expanduser("~/devel/results/directionality")
if not os.path.exists(RESULTS):
    os.mkdir(RESULTS)

NAMES = b.get_filenames()
N_NAMES = len(NAMES)

for i, name in enumerate(NAMES):
    print 'processing {} of {}: {}'.format(i, N_NAMES, name)
    img = b.get_cutout(name, rlevel=3)
    img = b.make_grey(img)
    magnitude, max_angles = t.directionality_filter(img)
    hsv_img = t.make_hsv(magnitude, max_angles)

    result_file = os.path.splitext(name)[0] + "_directionality.png"
    result_path = os.path.join(RESULTS, result_file)
    pp.imsave(result_path, hsv2rgb(hsv_img))
    
