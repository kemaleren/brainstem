import random
import numpy as np
import brainstem as b
import texture as t
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

names = b.get_filenames()
img = b.get_cutout(names[0], rlevel=2)
img = b.get_cutout(names[0], rlevel=3)
img = b.make_grey(img)
small_img = img[400:, 400:]
freqs = t.get_freqs(small_img)

kernels, all_freqs = t.make_filter_bank(freqs[-5:-2], np.deg2rad(np.arange(0, 180, 10)))
kernels = list(np.real(k) for k in kernels)

filtered, all_freqs = t.filter_image(small_img, kernels, all_freqs, select=False)

features = t.compute_features(filtered, all_freqs)
features = t.add_coordinates(features, spatial_importance=1)

pca = PCA(10)
X = features.reshape(-1, features.shape[-1])
X_t = pca.fit_transform(X)

# for visualization
sampled = np.vstack(random.sample(X_t, 5000))

model = MiniBatchKMeans(6)
model.fit(X_t)

f2 = np.power(filtered, 2)

angles = []
for i in range(18):
    angles.append(f2[:, :, i::18].sum(axis=2))

angles = np.dstack(angles)
idx = np.argmax(angles, axis=2)

x, y = np.indices(idx.shape)
f2_maxes = f2[x, y, idx]
ratio = f2_maxes / f2.mean(axis=2)
