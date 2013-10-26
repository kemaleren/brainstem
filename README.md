Brainstem project
=================

Some useful code for the brainstem project. Parts of this project will be incorporated into [this project](https://github.com/mistycheney/registration).

Right now there are two modules. The first reads images and thresholds them. It also optionally caches files as TIFs, which take up more space than JPEG2000 but are faster to read. Here is an example of how to use it:

    import brainstem

    all_data_files = brainstem.get_filenames()
    img = brainstem.get_cutout(all_data_files[0])
    sub_img = brainstem.random_image_sample(img)
    labeled_img = brainstem.segment_cells(sub_img, rgb=True)

You will have to modify ``brainstem.DATADIR`` to point to a directory containing jp2 files. If you do not want the caching functionality, set ``brainstem.USE_CACHE = False``.

The second module implements shape context descriptors, shape distance metrics via dynamic programming, and context-sensitive shape similarity via graph transduction. Here is an example of how to use it:

    import skimage.data
    import shape_context as sc

    # get a binary horse shape
    img = skimage.data.horse()
    img = skimage.color.rgb2gray(img)
    t = skimage.filter.threshold_otsu(img)
    binary_img = img < t

    # make a copy and cut off its legs
    legless = binary_img.copy()
    legless[200:, 200:] = 0

    # compute the distance between them
    sc.full_shape_distance(binary_img, legless)