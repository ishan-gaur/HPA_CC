"""
Grouping

What level of the directory hierarchy to group the cached images by. This is used to group images by well, plate, etc.
0: no grouping, all images should be aggregated into single tensors stored at the root of the dataset directory
1: group by well, images will be aggregated into the first level of subdirectories
etc.
If you want the grouping to be at the individual image level, select -1
"""
grouping = 1

"""
Channel Specification

channels
The names of the channels as they will appear in the dataset directory folders ie {channel}.png
This will be saved after you set it for the first time, so you only need to set it once.

cmaps
The colormaps to use for each channel. These are the names of the colormaps in the microfilm package.
Normally pure_{color} works fine.

dapi, tubl, calb2
The indices of the channels that correspond to DAPI, Tubulin, and Calb2 respectively in the channels
array. If you don't have calb2, set it to None. The others must be set to integers.
These are used by the segmentation code to feed in the correct channels to the segmentation model.
"""
channels=["blue", "red", "yellow", "green"]
cmaps=["pure_blue", "pure_red", "pure_yellow", "pure_green"]
dapi, tubl, calb2 = 0, 1, 2

"""
Image Normalization

norm_strategy
    min_max: normalize to [0, 1] using image min and max
    rescale: rescale to [0, 1] using datatype min and max
    threshold: threshold to [0, 1] by clipping everything outside the thresholds and rescaling
        must set norm_min and norm_max (None by default, should be integers)
    percentile: threshold to [0, 1] by clipping everything outside the percentiles and rescaling
        must set norm_min and norm_max (None by default, should be integers between 0 and 100)
"""
# norm_strategy = None
norm_strategy = 'min_max'
norm_min, norm_max = None, None
# norm_strategy = 'threshold'
# norm_min, norm_max = 500, 65535
# norm_strategy = 'percentile'
# norm_min, norm_max = 20, 99

"""
Clean Masks

rm_border
    True: remove cells whose nuclei touch the image border
remove_size: remove cells whose bounding box area is smaller than this value (integer)
"""
rm_border = True
remove_size = 1000

"""
Single Cell Data Creation (Cropping and Resizing)
cutoff: pixel length for the square crop
nuc_margin: minimum pixel margin around the nucleus after the crop (sometimes the nucleus
    will be outside the crop, which is originally centered around the bounding box)
output_image_size: pixel size of the resized image, same size means no resizing
"""
cutoff = 512
nuc_margin = 50
output_image_size = 512

"""
Sharpness Filtering
sharpness_threshold: threshold for sharpness filtering (float); if None, no filtering is done
"""
sharpness_threshold = None
