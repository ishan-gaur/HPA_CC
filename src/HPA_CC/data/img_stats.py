import torch
import numpy as np
from HPA_CC.data.pipeline import composite_images_from_paths
from HPA_CC.utils.img_tools import get_batch_percentiles, get_image_percentiles
from HPA_CC.utils.img_tools import min_max_normalization, rescale_normalization, threshold_normalization, percentile_normalization
from HPA_CC.utils.img_tools import sample_sharpness
from HPA_CC.data.data_viz import plot_intensities, barplot_percentiles, cdf_percentiles, histplot_percentiles
from HPA_CC.data.data_viz import save_image_grid, color_image_by_intensity, plot_hist_w_threshold
from HPA_CC.utils.pseudotime import intensities_to_pseudotime


def pixel_range_info(args, image_paths, CHANNELS, OUTPUT_DIR):
    # if args.calc_num < 500:
    #     print("Warning: using less than 500 images to calculate pixel range may result in inaccurate pixel range")
    image_sample_paths = np.random.choice(image_paths, args.calc_num)
    image_sample = composite_images_from_paths(image_sample_paths, CHANNELS)

    values, percentiles = get_batch_percentiles(image_sample, non_zero=False)
    thresholded_values, thresholded_percentiles = get_batch_percentiles(image_sample)

    barplot_percentiles(percentiles, values, CHANNELS, OUTPUT_DIR / 'pixel_percentiles.png')
    barplot_percentiles(thresholded_percentiles, thresholded_values, CHANNELS, OUTPUT_DIR / 'pixel_percentiles_non_zero.png')

    image_values, image_percentiles = get_image_percentiles(image_sample)
    # histplot_percentiles(image_percentiles, image_values, CHANNELS, OUTPUT_DIR / 'image_percentiles.png')
    cdf_percentiles(image_percentiles, image_values, CHANNELS, OUTPUT_DIR / 'image_percentiles_cdf.png')
        
def normalization_dry_run(args, config, image_paths, CHANNELS, OUTPUT_DIR, device, channel_slice=None):
    image_sample_paths = np.random.choice(image_paths, args.calc_num)
    image_sample = composite_images_from_paths(image_sample_paths, CHANNELS)

    if config.norm_strategy == 'min_max':
        norm_images, mins, maxes, intensities = min_max_normalization(image_sample)
    if config.norm_strategy == 'rescale':
        norm_images, mins, maxes, intensities = rescale_normalization(image_sample)
    if config.norm_strategy == 'threshold':
        assert config.norm_min is not None and config.norm_max is not None, "Must specify norm_min and norm_max in config for threshold normalization"
        norm_images, mins, maxes, intensities = threshold_normalization(image_sample, config.norm_min, config.norm_max)
    if config.norm_strategy == 'percentile':
        assert config.norm_min is not None and config.norm_max is not None, "Must specify norm_min and norm_max in config for percentile normalization"
        norm_images, mins, maxes, intensities = percentile_normalization(image_sample, config.norm_min, config.norm_max)

    if config.norm_strategy == 'threshold':
        strategy = f"{config.norm_strategy}_{config.norm_min}_{config.norm_max}"
    elif config.norm_strategy == 'percentile':
        strategy = f"{config.norm_strategy}_{config.norm_min}_{config.norm_max}"
    else:
        strategy = config.norm_strategy 
        
    intensity_hist_file = OUTPUT_DIR / 'image_intensity_histogram.png'
    print("Plotting image intensity histogram")
    plot_intensities(intensities, CHANNELS, intensity_hist_file, log=True)
    print(f"Saved image intensity histogram to {intensity_hist_file}")

    norm_intensities = norm_images.transpose(1, 0, 2, 3).reshape(len(CHANNELS), -1)
    norm_intensity_hist_file = OUTPUT_DIR / f'norm_{strategy}_image_intensity_histogram.png'
    print("Plotting normalized image intensity histogram")
    plot_intensities(norm_intensities, CHANNELS, norm_intensity_hist_file, log=True)
    print(f"Saved normalized image intensity histogram to {norm_intensity_hist_file}")

    mins, maxes = mins.squeeze(), maxes.squeeze()
    # print the 0, 25, 50, 75, 100 percentiles
    for channel in range(len(CHANNELS)):
        print(f"Channel {CHANNELS[channel]}")
        print("\tImage min and max percentiles:")
        print(f"\t{np.percentile(mins[channel], [0, 25, 50, 75, 100])}")
        print(f"\t{np.percentile(maxes[channel], [0, 25, 50, 75, 100])}")

    # save a grid of images before and after
    print("Saving image samples")
    norm_image_file = OUTPUT_DIR / f'normalized_images_{strategy}.png'
    image_sample = rescale_normalization(image_sample[:args.viz_num], stats=False)
    image_sample = torch.from_numpy(image_sample).to(device)
    norm_images = torch.from_numpy(norm_images[:args.viz_num]).to(device)
    images = torch.cat([image_sample, norm_images], dim=0)
    print(images.shape)
    if channel_slice is None:
        channel_slice = slice(0, len(CHANNELS))
    save_image_grid(images[:, channel_slice], norm_image_file, args.viz_num, config.cmaps[channel_slice])

def image_by_level_set(args, image_paths, CHANNELS, OUTPUT_DIR):
    image_sample_paths = np.random.choice(image_paths, args.viz_num)
    image_sample = composite_images_from_paths(image_sample_paths, CHANNELS)
    image_sample = rescale_normalization(image_sample, stats=False)
    color_image_by_intensity(image_sample, OUTPUT_DIR / 'intensity_colored_images.png')


def sharpness_dry_run(dataset_image_paths, sharpness_threshold, output_dir, cmaps):
    dataset_images = torch.cat([torch.load(p) for p in dataset_image_paths])
    sharpness = sample_sharpness(dataset_images)
    output_file = output_dir / f"sharpness_{sharpness_threshold}.png"
    plot_hist_w_threshold(sharpness, sharpness_threshold, output_file)
    filtered_out_images = dataset_images[sharpness < sharpness_threshold]
    assert len(filtered_out_images) > 0, "No images filtered out, try a higher threshold or check that you haven't \
        already filtered these images out. You may need to rebuild single_cell for this dataset with a lower threshold."
    print(f"Filtered out {len(filtered_out_images)} images")
    print(f"That is {len(filtered_out_images) / len(dataset_images) * 100:.2f}% of the dataset")
    nrow = int(len(filtered_out_images) ** 0.5)
    save_image_grid(filtered_out_images, output_dir / f"filtered_out_{sharpness_threshold}.png", nrow=nrow, cmaps=cmaps)

def new_well_fucci_stats(paths, gmnn_idx=2, cdt1_idx=3):
    image_path, mask_path = paths
    scope_name = image_path.parent.name.split('--')[0]
    sc_images = torch.load(image_path) # Cells x Channels x H x W
    sc_images = sc_images[:, [gmnn_idx, cdt1_idx]] # only calculating for GMNN and CDT1
    nuclei_masks = torch.load(mask_path) # Cells x H x W
    sc_nuclei = sc_images * nuclei_masks[:, None]
    mean_intensities = torch.sum(sc_nuclei, dim=(2, 3)) / torch.sum(nuclei_masks[:, None], dim=(2, 3)) # only calculating for GMNN and CDT1
    min_nonzero_GMNN = torch.min(mean_intensities[:, 0][mean_intensities[:, 0] > 0])
    min_nonzero_CDT1 = torch.min(mean_intensities[:, 1][mean_intensities[:, 1] > 0])
    log_mean_GMNN = torch.log(mean_intensities[:, 0] + min_nonzero_GMNN)
    log_mean_CDT1 = torch.log(mean_intensities[:, 1] + min_nonzero_CDT1)
    log_mean_fucci_intensities = torch.stack((log_mean_GMNN, log_mean_CDT1), dim=1)
    fucci_time, raw_time, well_std_int = intensities_to_pseudotime(log_mean_fucci_intensities.numpy())
    raw_time = raw_time * 2 * np.pi - np.pi
    return scope_name, len(sc_images), log_mean_fucci_intensities, well_std_int, fucci_time, raw_time

def well_fucci_stats(paths, gmnn_idx=2, cdt1_idx=3):
    image_path, mask_path = paths
    scope_name = image_path.parent.name.split('--')[0]
    sc_images = torch.load(image_path) # Cells x Channels x H x W
    nuclei_masks = torch.load(mask_path) # Cells x H x W
    sc_nuclei = sc_images * nuclei_masks[:, None]
    mean_intensities = torch.sum(sc_nuclei[:, 2:], dim=(2, 3)) / torch.sum(nuclei_masks[:, None], dim=(2, 3)) # only calculating for GMNN and CDT1
    min_nonzero_GMNN = torch.min(mean_intensities[:, 0][mean_intensities[:, 0] > 0])
    min_nonzero_CDT1 = torch.min(mean_intensities[:, 1][mean_intensities[:, 1] > 0])
    log_mean_GMNN = torch.log(mean_intensities[:, 0] + min_nonzero_GMNN)
    log_mean_CDT1 = torch.log(mean_intensities[:, 1] + min_nonzero_CDT1)
    log_mean_fucci_intensities = torch.stack((log_mean_GMNN, log_mean_CDT1), dim=1)
    fucci_time, raw_time, well_std_int = intensities_to_pseudotime(log_mean_fucci_intensities.numpy())
    raw_time = raw_time * 2 * np.pi - np.pi
    return scope_name, len(sc_images), log_mean_fucci_intensities, well_std_int, fucci_time, raw_time
