from tqdm import tqdm
from pathlib import Path
import torch
import numpy as np
from pipeline import load_index_paths, run_silent
from data_viz import save_image_grid

run_silent()

data_folder = Path("/data/ishang/FUCCI-dataset-well/")
index_file = "index_clean_no_border_rm_1000_sharp_1250.csv"
PRECALC_INDEX_PATH = data_folder / index_file

resolution = 8 # bit
buckets = 2 ** resolution
well_percentiles = np.load(data_folder / f"well_percentiles_{buckets}.npy")
normalized_well_percentiles = np.load(data_folder / "normalized_well_percentiles.npy")

def well_normalization_map(original_perc, transformed_perc, range_max=None):
    def normalize(well_images):
        # well images are B x C x H x W
        for c in range(original_perc.shape[0]):
            # bucket the image pixels into the percentile ranges
            perc_domain = np.concatenate([[0], original_perc[c]])
            perc_range = np.concatenate([[0], transformed_perc[c]])

            # then normalize the image by the percentile range using a linear extrapolation within the bucket
            def normalize_pixel(x, v):
                perc_below = perc_domain[x - 1]
                perc_above = perc_domain[x] if x < len(perc_domain) else perc_domain[-1]
                transformed_perc_below = perc_range[x - 1]
                transformed_perc_above = perc_range[x] if x < len(perc_range) else perc_range[-1]
                if perc_above == perc_below:
                    return np.random.uniform(transformed_perc_below, transformed_perc_above)
                bucket_pos = (v - perc_below) / (perc_above - perc_below)
                transformed_value = transformed_perc_below + bucket_pos * (transformed_perc_above - transformed_perc_below)
                return transformed_value

            normalize_pixel = np.vectorize(normalize_pixel)

            if range_max is None:
                well_images_perc = np.digitize(well_images[:, c], perc_domain, right=False)
                assert (well_images_perc > 1).any(), "No pixels in the image are nonzero"
                well_images[:, c] = normalize_pixel(well_images_perc.flatten(), well_images[:, c].flatten()).reshape(well_images.shape[0], well_images.shape[2], well_images.shape[3])
            else:
                assert range_max > 0, "range_max must be positive"
                assert type(range_max) == int, "range_max must be an integer"
                sample_points = np.linspace(0, range_max - 1, range_max).astype(well_images.dtype)
                sample_dig = np.digitize(sample_points, perc_domain, right=False)
                sample_norm = normalize_pixel(sample_dig, sample_points)
                well_images[:, c] = sample_norm[well_images[:, c].astype(int)] # will it let me do this?
        return well_images
    return normalize

if __name__ == "__main__":
    image_paths, cell_mask_paths, nuclei_mask_paths = load_index_paths(PRECALC_INDEX_PATH)
    mask_paths = cell_mask_paths

    normalized_image_paths = []
    for i, (image_path, mask_path) in tqdm(enumerate(zip(image_paths, mask_paths)), total=len(image_paths), desc="Calculating well percentiles"):
        normalization_function = well_normalization_map(well_percentiles[i], normalized_well_percentiles[i], range_max=(np.iinfo(np.uint16).max + 1))
        images = np.load(image_path).astype("float32")
        masks_path = str(mask_path) + ".npy"
        masks = np.load(masks_path)[:, None, ...].astype("float32")
        images = images * (masks > 0)
        images = images
        normalized_images = normalization_function(np.copy(images))
        new_image_path = image_path.parent / (image_path.stem + f"_spline_{buckets}_normalized.npy")
        np.save(new_image_path, normalized_images)
        normalized_image_paths.append(new_image_path)

    for orig, norm in zip(images, normalized_images):
        # normalize each channel of orig to [0, 1]
        orig = orig / orig.max()
        orig, norm = torch.from_numpy(orig.astype("float32")), torch.from_numpy(norm.astype("float32"))
        grid_images = torch.concatenate([orig, norm], dim=0)

        print(orig.min(), orig.max())
        print(torch.quantile(orig[orig > 0], torch.tensor([0, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99, 100]) / 100))
        print(norm.min(), norm.max())
        print(torch.quantile(norm[norm > 0], torch.tensor([0, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99, 100]) / 100))
        break
        
    grid_images = grid_images[:, None, ...]
    print(grid_images.shape)
    save_image_grid(grid_images, "test_image.png", nrow=orig.shape[0], cmaps=None)