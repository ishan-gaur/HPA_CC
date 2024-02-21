import cv2
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from microfilm.colorify import multichannel_to_rgb
from PIL import Image
from torchvision.utils import make_grid

silent = False

class Plot:
    def __init__(self, file_path):
        self.file_path = file_path

    def __enter__(self):
        plt.clf()

    def __exit__(self):
        plt.tight_layout()
        plt.savefig(self.file_path)
        plt.close()

def plot_intensities(intensities, channel_names, file_path, num_pixel_samples=100000, samples_per_bin=None, bins=100, sample=True, log=False, threshold=0):
    eps = 1e-10
    if sample:
        intensities = intensities[:, np.random.choice(
            intensities.shape[1], num_pixel_samples, replace=(num_pixel_samples > intensities.shape[1])
        )]
    if threshold is not None:
        thresholded_intensities = []
        for i in range(intensities.shape[0]):
            thresholded_intensities.append(intensities[i, intensities[i] > threshold])
        min_len = min([len(thresholded_intensities[i]) for i in range(len(thresholded_intensities))])
        for i in range(len(thresholded_intensities)):
            thresholded_intensities[i] = thresholded_intensities[i][:min_len]
        intensities = np.array(thresholded_intensities)
    if log:
        intensities = np.log10(intensities + eps)
    intensity_df = pd.DataFrame(intensities.transpose())
    intensity_df.columns = channel_names

    with Plot(file_path) as p:
        if samples_per_bin is not None:
            bins = num_pixel_samples // samples_per_bin
        sns.histplot(data=intensity_df, bins=bins)
        x_axis_info = f"(log 10 with eps {eps:.2E}{f' and thresholded at {threshold}' if threshold is not None else ''})"
        plt.xlabel(f'Intensity {x_axis_info if log else ""}')
        plt.ylabel(f'Number of pixels ({intensities.shape[1]} total)')

def barplot_percentiles(percentiles, values, channel_names, file_path, log=True):
    # because the stacked plot will add them one on top of the other
    for i in range(values.shape[0]):
        for j in range(values.shape[1] - 1, 0, -1): # go backwards so you don't overwrite the values you need
            values[i, j] = values[i, j] - values[i, j - 1]
    values_df = pd.DataFrame(values)
    percentiles = [f'{percentile}th percentile' for percentile in percentiles]
    values_df.columns = percentiles
    values_df.index = channel_names
    with Plot(file_path) as p:
        values_df.plot(kind='bar', stacked=True)
        plt.xlabel('Channel')
        plt.ylabel('Percentile Intensity')

def histplot_percentiles(percentiles, values, channel_names, file_path, log=False):
    values = values.transpose(2, 0, 1) # C x B x P -> P x C x B
    percentiles = [f'{percentile}th percentile' for percentile in percentiles]
    for percentile, data in zip(percentiles, values):
        samples_per_bin = 25
        assert values.shape[-1] > samples_per_bin, f'Not enough samples per bin ({samples_per_bin}) for {file_path.stem}_{percentile}.png'
        new_file_path = file_path.parent / (f'{file_path.stem}_{percentile}' + file_path.suffix)
        plot_intensities(data, channel_names, new_file_path, log=log, sample=False, samples_per_bin=samples_per_bin)

def cdf_percentiles(percentiles, values, channel_names, file_path, log=False):
    values = values.transpose(2, 0, 1) # C x B x P -> P x C x B
    percentiles = [f'{percentile}th percentile' for percentile in percentiles]
    for percentile, data in zip(percentiles, values):
        intensity_df = pd.DataFrame(data.transpose())
        intensity_df.columns = channel_names
        with Plot(new_file_path) as p:
            for channel in channel_names:
                sns.ecdfplot(data=intensity_df[channel], label=channel)
            plt.legend()
            plt.xlabel('Intensity')
            plt.ylabel('Cumulative Distribution')
            new_file_path = file_path.parent / (f'{file_path.stem}_{percentile}' + file_path.suffix)

def save_image(image, file_path, cmaps):
    # takes a torch tensor for the image
    if (image.shape[0] != 1 and image.shape[0] != 3) and cmaps is None:
        raise ValueError("Must specify cmaps if there are multiple channels")
    img = image.float()
    if not silent: print("Converting to RGB")
    if cmaps is not None:
        img, _, _, _ = multichannel_to_rgb(img.cpu().numpy(), cmaps=cmaps)
    elif type(img) == torch.Tensor:
        img = img.permute(1, 2, 0)
        img = img.cpu().numpy()
    img = Image.fromarray((255 * img[..., :3]).astype(np.uint8))
    img.save(file_path)
    if not silent: print(f"Saved image samples to {file_path}")
    return img

def save_image_grid(images, file_path, nrow, cmaps=None):
    images = images.float()
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(0, 1))
    save_image(grid, file_path, cmaps)

def color_image_by_intensity(images, file_path):
    # make a grid where the top row is the original image
    # each successive row is one of the channels, but colors by intensity
    # this will require making a color map for each channel thresholded at each intensity level
    # images is a tensor of shape (B, C, H, W)
    # color_maps = ["pure_red", "pure_orange", "pure_yellow", "pure_green", "pure_blue", "pure_purple", "pure_black"]
    color_maps = ["pure_red", "pure_green", "pure_blue"]
    # color_maps = ["Reds", "Oranges", "Greens", "Blues", "Purples", "Greys"]
    color_maps = color_maps[::-1]
    num_levels = len(color_maps)
    intensity_levels = np.logspace(0, 1, num_levels + 1) # +1 because we want to have each color map have a max and min threshold

    if not silent: print("Thresholding images")
    images = images.transpose(1, 0, 2, 3) # C x B x H x W
    intensity_mapped_channels = []
    for channel_images in images:
        thresholded_images = []
        for intensity in intensity_levels:
            thresholded_images.append((channel_images <= intensity))
        for i in range(len(thresholded_images) - 1, 0, -1):
            thresholded_images[i] &= (thresholded_images[i - 1] == False)
        # prod = np.prod(thresholded_images, axis=0)
        # assert np.all(prod == False), "Something went wrong with the thresholding"
        thresholded_images = np.stack(thresholded_images[1:]) # don't need the "zero" level
        assert np.all(np.sum(thresholded_images, axis=0) <= 1), "Something went wrong with the thresholding, some pixels are more than one color"
        assert np.any(np.sum(thresholded_images, axis=0) == 0), "Something went wrong with the thresholding, no pixels are zero"
        intensity_mapped_channels.append(thresholded_images)
    intensity_mapped_channels = np.stack(intensity_mapped_channels) # C x num_levels x B x H x W

    if not silent: print("Reshaping images")
    intensity_mapped_channels = np.array(intensity_mapped_channels) # C x num_levels x B x H x W
    intensity_mapped_channels = intensity_mapped_channels.transpose(0, 2, 1, 3, 4) # C x B x num_levels x H x W
    intensity_mapped_channels = intensity_mapped_channels.reshape(-1, num_levels, *images.shape[2:]) # C * B x num_levels x H x W
    # intensity_mapped_channels = intensity_mapped_channels.transpose(1, 0, 2, 3) # num_levels x C * B x H x W
    intensity_mapped_channels = torch.from_numpy(intensity_mapped_channels)

    if not silent: print("Saving images as grid")
    save_image_grid(intensity_mapped_channels, file_path, nrow=images.shape[1], cmaps=color_maps)

def plot_hist_w_threshold(metric, threshold, output_file):
    plt.clf()
    sns.histplot(metric.cpu().numpy(), bins=20)
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1)
    plt.savefig(output_file)
    plt.clf()

def plots_to_fig_grid(plot_figs):
    plot_images = []
    for plot in plot_figs:
        plot.canvas.draw()
        plot_images.append(np.frombuffer(plot.canvas.tostring_rgb(), dtype=np.uint8).reshape(plot.canvas.get_width_height()[::-1] + (3,)))
        plt.close(plot)
    
    plt.clf()
    plt.imshow(plot_images[0])
    fig = plt.figure()
    size = fig.get_size_inches()*fig.dpi
    plt.close()

    factor_w, factor_h = size / 100

    plt.clf()
    n_col = 5
    rows, cols = int(np.ceil(len(plot_images) / n_col)), n_col

    # for i in range(rows * cols - len(plot_images)):
    #     plot_images.append(np.ones_like(plot_images[0]))

    fig, axes = plt.subplots(rows, cols, figsize=(factor_w * cols, factor_h * rows), constrained_layout=True)
    for i, ax in enumerate(axes.flatten()):
        if i < len(plot_images):
            ax.imshow(plot_images[i])
        # ax.set_aspect('equal')
        ax.axis("off")
    plt.show()

def annotate_cell_image(rgb_image, masks, pseudotimes, phases):
    # image = cv2.imread(image_path)
    image = rgb_image
    for mask, time, phase in zip(masks, pseudotimes, phases):
        # Draw the bounding box on the image
        min_x = np.min(np.where(mask)[1])
        max_x = np.max(np.where(mask)[1])
        min_y = np.min(np.where(mask)[0])
        max_y = np.max(np.where(mask)[0])
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # Annotate with the classification score
        text = f"{time}: {phase:.2f}"
        cv2.putText(image, text, (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image