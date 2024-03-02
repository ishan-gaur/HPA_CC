import torch
import numpy as np
from kornia.filters import sobel

silent = False

def min_max_normalization(images, stats=True, non_zero=True):
    # images come in as B x C x H x W
    images = images.transpose(1, 0, 2, 3)
    mins, maxes, intensities = get_min_max_int(images)
    if non_zero:
        min_nonzero = [intensities[channel][intensities[channel] > 0].min() for channel in range(len(intensities))]
        min_nonzero = np.array(min_nonzero).reshape(mins.shape)
        mins = min_nonzero
    if not silent: print("Normalizing images")
    images = np.clip(images, mins, maxes)
    norm_images = (images - mins) / (maxes - mins)
    # convert norm images back to B x C x H x W
    norm_images = norm_images.transpose(1, 0, 2, 3)
    norm_images = norm_images.astype(np.float32)
    if not stats:
        return norm_images
    # intensities for all images for a given channel
    return norm_images, mins, maxes, intensities

def get_min_max_int(images):
    # images are now C x B x H x W
    num_channels = images.shape[0]
    if not silent: print("Calculating image min and max")
    mins = images.min(axis=(1, 2, 3), keepdims=True)
    maxes = images.max(axis=(1, 2, 3), keepdims=True)
    intensities = images.reshape(num_channels, -1)
    return mins, maxes, intensities

def percentile_normalization(images, perc_min, perc_max, stats=True):
    if not silent: print("Calculating image percentiles")
    percentiles, _ = get_images_percentiles(images, percentiles=[perc_min, perc_max]) # C x P
    print(percentiles)
    percentiles = percentiles[None, ...] # add batch dimension
    min_int, max_int = percentiles[..., 0], percentiles[..., 1]
    return threshold_normalization(images, min_int[..., None, None], max_int[..., None, None], stats=stats)

def threshold_normalization(images, min_int, max_int, stats=True):
    norm_images = np.clip(images, min_int, max_int)
    norm_images = (norm_images - min_int) / (max_int - min_int)
    norm_images = norm_images.astype(np.float32)
    if not stats:
        return norm_images
    mins, maxes, intensities = get_min_max_int(images)
    return norm_images, mins, maxes, intensities

def rescale_normalization(images, stats=True):
    dtype_max = np.iinfo(images.dtype).max
    if not silent: print("Normalizing images")
    norm_images = images / dtype_max
    norm_images = norm_images.astype(np.float32)
    if not stats:
        return norm_images
    mins, maxes, intensities = get_min_max_int(norm_images)
    return norm_images, mins, maxes, intensities

def get_images_percentiles(images, percentiles=[90, 99, 99.99], non_zero=True):
    # returns a list of percentiles for each batch by channel: C x P
    num_channels = images.shape[1]
    channel_images = images.transpose(1, 0, 2, 3).reshape(num_channels, -1)
    if not silent: print("Calculating well pixel percentiles")
    if non_zero:
        channel_images = [channel_pixels[channel_pixels > 0] for channel_pixels in channel_images]
    values = np.array([np.percentile(channel_pixels, percentiles) for channel_pixels in channel_images])
    return values, percentiles

def image_cells_sharpness(image, cell_mask):
    if len(image.shape) != 3 or len(cell_mask.shape) != 2:
        raise ValueError(f"This method only takes single images. Input image must be of shape C x H x W and cell_mask must be of shape H x W.\
                         image {image.shape} and mask {cell_mask.shape} were given.")
    image = image[None, ...] # kornia expects a batch dimension
    image_sharpness = sobel(image)
    # filling in None in case the cell masks aren't consecutive
    cell_mask = cell_mask.astype(int)
    sharpness_levels = [None for _ in range(cell_mask.max() + 1)]
    for cell in np.unique(cell_mask):
        mask_tile = list(image.shape)
        mask_tile[-cell_mask.ndim:] = [1] * cell_mask.ndim
        image_mask = torch.tensor(cell_mask == cell).tile(mask_tile)
        cell_sharpness = image_sharpness[image_mask]
        if len(cell_sharpness) == 0:
            continue
        sharpness_levels[cell] = cell_sharpness.std()
    sharpness_levels[0] = None # since the 0th "cell" is just the background
    assert len(sharpness_levels) == cell_mask.max().astype(int) + 1
    return sharpness_levels

def sample_sharpness(images):
    from kornia.filters import sobel
    image_sharpness = sobel(images)
    image_sharpness = image_sharpness.std(dim=(1,2,3))
    return image_sharpness

def get_batch_percentiles(images, percentiles=[0, 25, 50, 90, 99, 99.99], non_zero=True):
    num_channels = images.shape[1]
    channel_pixels = images.transpose(1, 0, 2, 3).reshape(num_channels, -1)
    if not silent: print("Calculating dataset pixel percentiles")
    if non_zero:
        channel_pixels = [pixels[pixels > 0] for pixels in channel_pixels]
    values = []
    for c, pixels in enumerate(channel_pixels):
        if len(pixels) == 0:
            print(f"Warning: Channel {c} has no {'non-zero ' if non_zero else ''}pixels")
            values.append([1.0 for _ in percentiles])
        else:
            values.append(np.percentile(pixels, percentiles).tolist())
    values = np.array(values)
    return values, percentiles

def get_image_percentiles(images, percentiles=[90, 99, 99.99], non_zero=True):
    # returns a list of percentiles for each image by channel: C x B x P
    num_channels, num_images = images.shape[1], images.shape[0]
    channel_images = images.transpose(1, 0, 2, 3).reshape(num_channels, num_images, -1)
    if not silent: print("Calculating image pixel percentiles")
    if non_zero:
        channel_images = [[image[image > 0] for image in images] for images in channel_images]
    values = np.array([[np.percentile(image, percentiles) for image in images] for images in channel_images])
    return values, percentiles

# def get_image_percentiles(images, percentiles=[90, 99, 99.99], non_zero=True):
#     num_channels, num_images = images.shape[1], images.shape[0]
#     channel_images = images.transpose(1, 0, 2, 3).reshape(num_channels, num_images, -1)
#     if not silent: print("Calculating image pixel percentiles")
#     if non_zero:
#         channel_images = np.where(channel_images > 0, channel_images, np.nan)
#     values = np.nanpercentile(channel_images, percentiles, axis=2)
#     return values, percentiles