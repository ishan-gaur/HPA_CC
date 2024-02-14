import shutil
import sys
import inspect
import argparse
import pickle as pkl
from pathlib import Path
from importlib import import_module
from tqdm import tqdm

from HPA_CC.data.dataset import CellImageDataset, intensity_data_name, cls_embedding_name
from HPA_CC.data.dataset import angle_label_name, pseudotime_label_name, phase_label_name
import HPA_CC.data.pipeline as pipeline
from HPA_CC.data.pipeline import create_image_paths_file, image_paths_from_folders, create_data_path_index, load_index_paths, load_channel_names, save_channel_names
from HPA_CC.data.pipeline import segmentator_setup, get_masks, normalize_images, filter_masks_by_sharpness, clean_and_save_masks, crop_images, resize
import HPA_CC.data.well_normalization as spline
from HPA_CC.data.well_normalization import buckets
from HPA_CC.data.img_stats import pixel_range_info, normalization_dry_run, image_by_level_set, sharpness_dry_run
from HPA_CC.utils.img_tools import get_batch_percentiles
from HPA_CC.models.dino import run_silent as dino_silent

import torch
import numpy as np
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

from config import OUTPUT_DIR

stats_opt = ['norm', 'pix_range', 'int_img', 'sample', 'sharp', 'name']
stats_opt_desc = {
    'norm': 'Normalize images and show results and statistics',
    'pix_range': 'Show pixel range statistics, esp percentile intensities per channel',
    'int_img': 'Show image level intensity statistics',
    'sample': 'Show sample images from final dataset',
    'sharp': 'Show sharpness statistics',
    'name': 'Show dataset names'
}
stats_desc = '\n'.join([f"{opt}: {stats_opt_desc[opt]};" for opt in stats_opt])
NORM, PIX_RANGE, INT_IMG, SAMPLE, SHARP, NAME = 0, 1, 2, 3, 4, 5

parser = argparse.ArgumentParser(description='Dataset preprocessing pipline')
parser.add_argument('--data_dir', type=str, help='Path to dataset, should be absolute path', required=True)
parser.add_argument('--output_dir', type=str , help='Path to output directory, should be absolute path')
parser.add_argument('--config', type=str, help='Path to config file, should be absolute path')
parser.add_argument('--name', type=str, help='Name of dataset version to look up in dataset folder (used for cached results)', default='unspecified')
parser.add_argument('--stats', type=str, help=f"Image stats to show, options include: {stats_opt}\n{stats_desc}", choices=stats_opt)
parser.add_argument('--viz_num', type=int, default=5, help='Number of samples to show')
parser.add_argument('--calc_num', type=int, default=30, help='Number of samples to use for calculating image stats')
parser.add_argument('--all', action='store_true', help='Run all steps')
parser.add_argument('--image_mask_cache', action='store_true', help='Save images')
parser.add_argument('--clean_masks', action='store_true', help='Clean masks: remove small objects and join cells without nuclei, etc.')
parser.add_argument('--filter_sharpness', action='store_true', help='Filter out blurry images based on config sharpness threshold')
parser.add_argument('--normalize', action='store_true', help='Normalize images')
parser.add_argument('--single_cell', action='store_true', help='Crop and save single cell images')
parser.add_argument('--rgb', action='store_true', help='Convert images to RGB')
parser.add_argument('--dinov2', action='store_true', help='Cache dinov2 cls embeddings')
parser.add_argument('--dino_hpa', action='store_true', help='Cache DINO HPA cls embeddings on reference channels only')
parser.add_argument('--int_dist', action='store_true', help='Concatenate intensity distributions to embeddings')
parser.add_argument('--fucci_gmm', action='store_true', help='Fit GMM to FUCCI intensities')
parser.add_argument('--labels', action='store_true', help='Calculate angular, pseudotime, and phase class labels for the samples')
parser.add_argument('--scope', action='store_true', help='Calculate phase labels using a GMM trained at the microscope level')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for dino inference')
parser.add_argument('--device', type=int, default=7, help='GPU device number')
parser.add_argument('--rebuild', action='store_true', help='Rebuild specifed steps even if files exist')
parser.add_argument('--save_samples', action='store_true', help='Save sample outputs for each well')
parser.add_argument('--silent', action='store_true', help='Run pipeline in silent mode')

args = parser.parse_args()

#===================================================================================================
# Basic Setup
#===================================================================================================
if args.silent:
    pipeline.suppress_warnings = True
    pipeline.run_silent()
    dino_silent()
else:
    print("Running in verbose mode")

DATA_DIR = Path(args.data_dir)
if not DATA_DIR.is_absolute():
    DATA_DIR = Path.cwd() / DATA_DIR
    print(f"Converted relative path to absolute path: {DATA_DIR}")
if not DATA_DIR.exists():
    raise ValueError(f"Data directory {DATA_DIR} does not exist")

OUTPUT_DIR = Path(args.output_dir) if args.output_dir is not None else OUTPUT_DIR
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)
    print(f"Created output directory {OUTPUT_DIR}")

args.config = Path(args.config)
if not args.config.is_absolute():
    args.config = Path.cwd() / args.config
if not args.config.exists():
    raise ValueError(f"Config file {args.config} does not exist")
sys.path.append(str(args.config.parent))
config = import_module(str(args.config.stem))

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

#===================================================================================================
# Set up paths for data and results--NEEDS TO BE UPDATED for any new functionality added
#                                    -------------------
#===================================================================================================
# just a list of the image and mask files
no_name = (args.name == 'unspecified')
BASE_INDEX = DATA_DIR / "index.csv"

# index after the cleaning step--remove small objects and partial cells
CLEAN_SUFFIX = f"{'no_border_' if config.rm_border else ''}{'rm_' + str(config.remove_size)}"
CLEAN_INDEX = DATA_DIR / f"index_clean_{CLEAN_SUFFIX}.csv"

# index after the sharpness filtering step--remove blurry images
SHARP_SUFFIX = f"{config.sharpness_threshold}".split(".")[-1] if config.sharpness_threshold is not None else "none"
SHARP_INDEX = DATA_DIR / f"{CLEAN_INDEX.stem}_sharp_{SHARP_SUFFIX}.csv"

# index after intensity normalization
if config.norm_strategy in ['threshold', 'percentile']:
    NORM_SUFFIX = f"{config.norm_strategy}{f'_{config.norm_min}_{config.norm_max}' if config.norm_strategy in ['threshold', 'percentile'] else ''}"
elif config.norm_strategy == 'spline':
    if args.image_mask_cache or args.all:
        print("WARNING: automatically setting buckets to 2^8 for spline normalization")
        buckets = 2 ** 8
    else:
        from HPA_CC.data.well_normalization import buckets
    NORM_SUFFIX = f"{config.norm_strategy}_{buckets}"
else:
    NORM_SUFFIX = config.norm_strategy
NORM_INDEX = DATA_DIR / f"{CLEAN_INDEX.stem if config.sharpness_threshold is None else SHARP_INDEX.stem}_norm_{NORM_SUFFIX}.csv"

# index after cropping single cell images (this is the output imaging dataset)
NAME_INDEX = DATA_DIR / f"index_{args.name}.csv"
CONFIG_FILE = DATA_DIR / f"{args.name}.py"

try:
    sys.path.append(str(CONFIG_FILE.parent))
    dataset_config = None if no_name else import_module(str(CONFIG_FILE.stem))
except ModuleNotFoundError:
    if NAME_INDEX.exists():
        print(f"Config file {CONFIG_FILE} not found, but index file {NAME_INDEX} exists, this should not be the case")
    dataset_config = None

# indices for the actual ML datasets
RGB_DATASET = DATA_DIR / f"rgb_{args.name}.pt"
if not any([args.dinov2, args.dino_hpa, args.all]):
    EMBEDDINGS_DATASET = None
else:
    EMBEDDINGS_DATASET = cls_embedding_name(DATA_DIR, args.name, hpa=(not args.dinov2))

INT_DATASET = intensity_data_name(DATA_DIR, args.name)

DINO_CONFIG = Path.cwd() / "configs" / "dino_config.yaml"

# GMM_PATH = DATA_DIR / f"gmm_{args.name}.pkl"
# GMM_PROBS = DATA_DIR / f"gmm_probs_{args.name}.pt"
ANGULAR_LABELS = angle_label_name(DATA_DIR, args.name)
PSEUDOTIME_LABELS = pseudotime_label_name(DATA_DIR, args.name)
PHASE_LABELS = phase_label_name(DATA_DIR, args.name, args.scope)

CHANNELS = load_channel_names(DATA_DIR) if config.channels is None else config.channels
if config.channels is None:
    save_channel_names(DATA_DIR, CHANNELS)
DAPI, TUBL, CALB2 = config.dapi, config.tubl, config.calb2

#===================================================================================================
# Image stats are useful for exploring the data and trying out parameters for the pipeline config
#===================================================================================================
if args.stats is not None:
    if args.stats == stats_opt[NAME]:
        # find all files like index_{name}.csv and print name
        print(f"Dataset names:")
        for file in DATA_DIR.glob("index_*.csv"):
            print(f"{file.stem[6:]}")

    data_paths_file, num_paths = create_image_paths_file(DATA_DIR, level=config.grouping)
    image_paths = image_paths_from_folders(data_paths_file)
    if args.stats == stats_opt[PIX_RANGE]:
        pixel_range_info(args, image_paths, CHANNELS, OUTPUT_DIR)
    if args.stats == stats_opt[NORM]:
        normalization_dry_run(args, config, image_paths, CHANNELS, OUTPUT_DIR, device)
    if args.stats == stats_opt[INT_IMG]:
        image_by_level_set(args, image_paths, CHANNELS, OUTPUT_DIR)
    if args.stats == stats_opt[SAMPLE]:
        from HPA_CC.data.data_viz import save_image_grid, save_image
        assert not no_name, "Name of dataset must be specified"
        assert dataset_config is not None, "Dataset config file must be specified via name, this means that the config for this data doesn't exist or doesn't make the provided name"
        assert BASE_INDEX.exists() and NAME_INDEX.exists(), "Index files do not exist, run pipeline with at least until --single_cell"
        ORIGINAL_IMG = OUTPUT_DIR / "original_image.png"
        DATASET_IMG = OUTPUT_DIR / "dataset_cells.png"
        image_paths, _, _ = load_index_paths(BASE_INDEX)
        og_image_paths = image_paths[:args.viz_num]
        dataset_image_paths, _, _ = load_index_paths(NAME_INDEX)
        dataset_image_paths = dataset_image_paths[:args.viz_num]
        for i, (image_path, cell_images_path) in enumerate(zip(og_image_paths, dataset_image_paths)):
            image = torch.tensor(np.load(image_path).astype(np.float32)).squeeze()
            cell_images = torch.load(cell_images_path)
            img_file = ORIGINAL_IMG.with_name(f"{ORIGINAL_IMG.stem}_{i}{ORIGINAL_IMG.suffix}")
            cell_file = DATASET_IMG.with_name(f"{DATASET_IMG.stem}_{i}{DATASET_IMG.suffix}")
            save_image(image, img_file, cmaps=dataset_config.cmaps)
            save_image_grid(cell_images, cell_file, nrow=5, cmaps=dataset_config.cmaps)
    if args.stats == stats_opt[SHARP]:
        assert not no_name, "Name of dataset must be specified"
        assert dataset_config is not None, "Dataset config file must be specified via name, this means that the config for this data doesn't exist or doesn't make the provided name"
        assert NAME_INDEX.exists(), "Index files do not exist, run pipeline with at least until --single_cell"
        assert config.sharpness_threshold is not None, "Sharpness threshold must be specified in config file"
        dataset_image_paths, _, _ = load_index_paths(NAME_INDEX)
        dataset_image_paths = dataset_image_paths[:args.calc_num]
        sharpness_dry_run(dataset_image_paths, config.sharpness_threshold, OUTPUT_DIR, dataset_config.cmaps)

#===================================================================================================
# Execution of Pipeline Steps (most implementation details are in the HPA_CC.data.pipeline module)
# Each step works by checking if the necessary precursor files exist
# Then ensures the command line arguments make sense
# Pipeline commands called here all take in index files, which are the central object of the pipeline
# The index files keep track of results of each intermediate step in terms of the resulting images
# and segmentation masks. The files are named according to the series of transformations that have
# been applied to them.
# Pipeline steps output lists of these filenames, which can then be combined as desired to create 
# a new index file.
# The pipeline outputs "named" index files. These names identify a unique "dataset" that can be 
# read out from the original files. The names also correspond to a copy of the generating config
# that is copied to the dataset directory for reproducibility/debugging later on.
#===================================================================================================

if args.image_mask_cache or args.all:
    print("Caching composite images and getting segmentation masks")
    data_paths_file, num_paths = create_image_paths_file(DATA_DIR, level=config.grouping, overwrite=args.rebuild)
    image_paths = image_paths_from_folders(data_paths_file)
    if BASE_INDEX.exists() and not args.rebuild:
        print("Index file already exists, skipping. Set --rebuild to overwrite.")
        save_channel_names(DATA_DIR, CHANNELS)
    else:
        multi_channel_model = True if CALB2 is not None else False
        segmentator = segmentator_setup(multi_channel_model, device)
        image_paths, nuclei_mask_paths, cell_mask_paths = get_masks(segmentator, image_paths, CHANNELS, DAPI, TUBL, CALB2, rebuild=args.rebuild)
        create_data_path_index(image_paths, cell_mask_paths, nuclei_mask_paths, BASE_INDEX, overwrite=True)
        save_channel_names(DATA_DIR, CHANNELS)

if args.clean_masks or args.all:
    print("Cleaning masks")
    if CLEAN_INDEX.exists() and not args.rebuild:
        print("Index file already exists, skipping. Set --rebuild to overwrite.")
    else:
        assert BASE_INDEX.exists(), "Index file does not exist, run --image_mask_cache first"
        image_paths, cell_mask_paths, nuclei_mask_paths = load_index_paths(BASE_INDEX)
        clean_cell_mask_paths, clean_nuclei_mask_paths, num_original, num_removed = clean_and_save_masks(cell_mask_paths, nuclei_mask_paths, CLEAN_SUFFIX,
            rm_border=config.rm_border, remove_size=config.remove_size)
        create_data_path_index(image_paths, clean_cell_mask_paths, clean_nuclei_mask_paths, CLEAN_INDEX, overwrite=True)
        print("Fraction removed:", num_removed / num_original)
        print("Total cells removed:", num_removed)
        print("Total cells remaining:", num_original - num_removed)

if args.filter_sharpness or args.all:
    print("Filtering masks by sharpness")
    if SHARP_INDEX.exists() and not args.rebuild:
        print("Index file already exists, skipping. Set --rebuild to overwrite.")
    else:
        assert CLEAN_INDEX.exists(), "Index file does not exist, run --clean_masks first"
        clean_image_paths, clean_cell_mask_paths, clean_nuclei_mask_paths = load_index_paths(CLEAN_INDEX)
        # we just overwrite the segmentation masks with the filtered ones so no need to get new paths
        sharp_cell_mask_paths, sharp_nuclei_mask_paths, num_removed, num_total = filter_masks_by_sharpness(clean_image_paths, 
            clean_cell_mask_paths, clean_nuclei_mask_paths, config.sharpness_threshold, config.dapi, config.tubl, SHARP_SUFFIX,
            args.save_samples, config.cmaps)
        create_data_path_index(clean_image_paths, sharp_cell_mask_paths, sharp_nuclei_mask_paths, SHARP_INDEX, overwrite=True)
        print("Fraction blurry that were removed:", num_removed / num_total)

if args.normalize or args.all:
    print("Normalizing images")
    if NORM_INDEX.exists() and not args.rebuild:
        print("Index file already exists, skipping. Set --rebuild to overwrite.")
    else:
        assert SHARP_INDEX.exists() or CLEAN_INDEX.exists(), "Index file does not exist, run --image_mask_cache (and optionally --clean_masks) first"
        if config.sharpness_threshold is not None:
            print("Using sharpness filtered images")
            SRC_INDEX = SHARP_INDEX
        else:
            print("Using cleaned images")
            SRC_INDEX = CLEAN_INDEX
        assert SRC_INDEX.exists(), "Index file does not exist, run --filter_sharpness (and optionally --clean_masks) first"

        if config.norm_strategy is None:
            shutil.copy(SRC_INDEX, NORM_INDEX)
        elif config.norm_strategy == 'spline':
            if SRC_INDEX != spline.PRECALC_INDEX_PATH:
                raise NotImplementedError("Spline normalization requires precalculated well percentiles, run well_percentiles.ipynb and well_normalization.py first with your desired input data.")
            image_paths, cell_mask_paths, nuclei_mask_paths = load_index_paths(spline.PRECALC_INDEX_PATH)
            mask_paths = cell_mask_paths
            normalized_image_paths = []
            for i, (image_path, mask_path) in tqdm(enumerate(zip(image_paths, mask_paths)), total=len(image_paths), desc="Calculating well percentiles"):
                normalization_function = spline.well_normalization_map(spline.well_percentiles[i], spline.normalized_well_percentiles[i], range_max=(np.iinfo(np.uint16).max + 1))
                images = np.load(image_path).astype("float32")
                masks_path = str(mask_path) + ".npy"
                masks = np.load(masks_path)[:, None, ...].astype("float32")
                images = images * (masks > 0)
                images = images
                normalized_images = normalization_function(np.copy(images))
                new_image_path = image_path.parent / (image_path.stem + f"_spline_{spline.buckets}_normalized")
                np.save(new_image_path, normalized_images)
                new_image_path = new_image_path.parent / (new_image_path.stem + ".npy")
                normalized_image_paths.append(new_image_path)
            create_data_path_index(normalized_image_paths, cell_mask_paths, nuclei_mask_paths, NORM_INDEX, overwrite=True)
        else:
            image_paths, cell_mask_paths, nuclei_mask_paths = load_index_paths(SRC_INDEX)
            norm_paths = normalize_images(image_paths, cell_mask_paths, config.norm_strategy, config.norm_min, config.norm_max,
                NORM_SUFFIX, batch_size=100 if config.grouping == -1 else 1, save_samples=args.save_samples, cmaps=config.cmaps)
            create_data_path_index(norm_paths, cell_mask_paths, nuclei_mask_paths, NORM_INDEX, overwrite=True)

if args.single_cell or args.all:
    print("Cropping single cell images")
    assert not no_name, "Name of dataset must be specified"
    if NAME_INDEX.exists() and not args.rebuild:
        print("Index file already exists, skipping. Set --rebuild to overwrite.")
    else:
        assert NORM_INDEX.exists(), "Index file for normalized images does not exist, run --normalize first"
        image_paths, cell_mask_paths, nuclei_mask_paths = load_index_paths(NORM_INDEX)
        seg_image_paths, clean_cell_mask_paths, clean_nuclei_mask_paths = crop_images(image_paths, cell_mask_paths, nuclei_mask_paths, config.cutoff, config.nuc_margin)
        final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths = resize(seg_image_paths, clean_cell_mask_paths, clean_nuclei_mask_paths, config.output_image_size, args.name)
        create_data_path_index(final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths, NAME_INDEX, overwrite=True)

        # save the source of the config module to the data directory with name args.name + '.py'
        # this will allow us to reproduce the results later
        with open(CONFIG_FILE, "w") as f:
            f.write("\n\n# Source of config module:\n")
            f.write(f"\n\n# Using normalized images from {NORM_INDEX}:\n")
            f.write(inspect.getsource(config))

        dataset_config = config

# if args.rgb or args.all:
#     assert not no_name, "Name of dataset must be specified"
#     assert dataset_config is not None, "Dataset config file must be specified via name, this means that the config for this data doesn't exist or doesn't make the provided name"
#     if SimpleDataset.has_cache_files(RGB_DATASET) and not args.rebuild:
#         print("RGB images file already exists, skipping. Set --rebuild to overwrite.")
#     else:
#         print("Creating RGB images")
#         assert NAME_INDEX.exists(), "Index file for single cell images does not exist, run --single_cell first"
#         assert CONFIG_FILE.exists(), "Config file does not exist for the dataset, something might have gone wrong when you ran --single_cell"
#         dataset = CellImageDataset(NAME_INDEX, dataset_config.cmaps, batch_size=args.batch_size)
#         rgb_dataset = dataset.as_rgb()
#         rgb_dataset.save(RGB_DATASET)

if args.int_dist or args.all:
    print("Caching well intensity distributions for each sample")
    if INT_DATASET.exists() and not args.rebuild:
        print("Intensity distributions file already exists, skipping. Set --rebuild to overwrite.")
    else:
        assert not no_name, "Name of dataset must be specified"
        assert dataset_config is not None, "Dataset config file must be specified via name, this means that the config for this data doesn't exist or doesn't make the provided name"
        assert NAME_INDEX.exists(), "Index file for single cell images does not exist, run --single_cell first"
        print("Concatenating intensity distributions to embeddings")
        channels = [dataset_config.dapi, dataset_config.tubl]
        dataset = CellImageDataset(NAME_INDEX, channels=channels, batch_size=args.batch_size)
        well_iterator = iter(dataset)
        percentiles = np.linspace(0, 100, buckets)
        well_percentiles = []
        for well_images in tqdm(well_iterator, total=dataset.iter_len(), desc="Getting intensity distributions"):
            well_int_levels, _ = get_batch_percentiles(well_images.cpu().numpy(), percentiles)
            well_percentiles.append(np.array([well_int_levels for _ in range(len(well_images))]))
        well_percentiles = np.concatenate(well_percentiles)
        well_percentiles = np.concatenate((well_percentiles[:, 0], well_percentiles[:, 1]), axis=1)
        well_percentiles = torch.tensor(well_percentiles)
        torch.save(well_percentiles, INT_DATASET)
        print(well_percentiles.shape)

if args.dinov2:
    from HPA_CC.models.dino import DINO
    print("Caching dinov2 cls embeddings")
    assert not no_name, "Name of dataset must be specified"
    # do I really need the next line at this point?
    assert dataset_config is not None, "Dataset config file must be specified via name, this means that the config for this data doesn't exist or doesn't make the provided name"

    if EMBEDDINGS_DATASET.exists() and not args.rebuild:
        print("Embeddings file already exists, skipping. Set --rebuild to overwrite.")
    else:
        assert NAME_INDEX.exists(), "Index file for single cell image dataset does not exist, run --single_cell first"
        print("Running DINO model to get embeddings")
        if type(dataset_config.output_image_size) != tuple:
            dataset_config.output_image_size = (dataset_config.output_image_size, dataset_config.output_image_size)
        dino = DINO(imsize=dataset_config.output_image_size).to(device)
        channels = [dataset_config.dapi, dataset_config.tubl, None]
        dataset = CellImageDataset(NAME_INDEX, channels=channels, batch_size=args.batch_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, shuffle=False)
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(iter(dataloader), desc="Embedding images with DINOv2"):
                batch = batch.to(device)
                batch_embedding = dino(batch).cpu()
                embeddings.append(batch_embedding)
        embeddings = torch.cat(embeddings)
        torch.save(embeddings, EMBEDDINGS_DATASET)
        print(f"Saved embeddings with shape {embeddings.shape} at {EMBEDDINGS_DATASET}")

elif args.dino_hpa or args.all:
    from HPA_CC.models.dino import DINO_HPA
    print("Caching DINO HPA cls embeddings")
    assert not no_name, "Name of dataset must be specified"
    # do I really need the next line at this point?
    assert dataset_config is not None, "Dataset config file must be specified via name, this means that the config for this data doesn't exist or doesn't make the provided name"

    if device != "cuda:0":
        print("Warning: DINO HPA model uses DataParallel, which requires the module to be on cuda:0, moving to cuda:0")
        device = "cuda:0"

    if EMBEDDINGS_DATASET.exists() and not args.rebuild:
        print("Embeddings file already exists, skipping. Set --rebuild to overwrite.")
    else:
        assert NAME_INDEX.exists(), "Index file for single cell image dataset does not exist, run --single_cell first"
        print("Running DINO model to get embeddings")
        if type(dataset_config.output_image_size) != tuple:
            dataset_config.output_image_size = (dataset_config.output_image_size, dataset_config.output_image_size)
        dino = DINO_HPA(DINO_CONFIG, device=device)
        channels = [dataset_config.dapi, dataset_config.tubl]
        dataset = CellImageDataset(NAME_INDEX, channels=channels, batch_size=args.batch_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, shuffle=False)
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(iter(dataloader), desc="Embedding images with HPA DINO"):
                batch = batch.to(device)
                batch_embedding = dino.predict_cls_ref_concat(batch).cpu()
                embeddings.append(batch_embedding)
        embeddings = torch.cat(embeddings)
        torch.save(embeddings, EMBEDDINGS_DATASET)
        print(f"Saved embeddings with shape {embeddings.shape} at {EMBEDDINGS_DATASET}")

# if args.fucci_gmm or args.all:
#     from sklearn.mixture import GaussianMixture
#     from HPA_CC.data.dataset import CellImageDataset, SimpleDataset
#     assert not no_name, "Name of dataset must be specified"
#     assert NAME_INDEX.exists(), "Index file for single cell images does not exist, run --single_cell first"
#     if GMM_PROBS.exists() and not args.rebuild:
#         print("GMM probabilities file already exists, skipping. Set --rebuild to overwrite.")
#     else:
#         dataset = CellImageDataset(NAME_INDEX)
#         dataloader = DataLoader(dataset, batch_size=1000, num_workers=1, shuffle=False)
#         FUCCI_intensities = []
#         for batch in tqdm(iter(dataloader), desc="Getting FUCCI intensities"):
#             FUCCI_intensities.append(torch.mean(batch[:, 2:], dim=(2, 3)))
#         FUCCI_intensities = torch.cat(FUCCI_intensities)
#         FUCCI_intensities = torch.log10(FUCCI_intensities + 1e-6)
#         plt.clf()
#         sns.kdeplot(x=FUCCI_intensities[:, 0], y=FUCCI_intensities[:, 1])
#         plt.savefig(DATA_DIR / f"fucci_plot_{args.name}.png")
#         plt.clf()

#         print("Creating GMM")
#         gmm = GaussianMixture(n_components=3)
#         gmm.fit(FUCCI_intensities)
#         pickle.dump(gmm, open(GMM_PATH, "wb"))
#         print("Saved GMM to pickle file at " + str(GMM_PATH))

#         print("Creating GMM probabilities")
#         probs = gmm.predict_proba(FUCCI_intensities)
#         probs = torch.tensor(probs)
#         torch.save(probs, GMM_PROBS)
#         print("Saved GMM probabilities to torch .pt file at " + str(GMM_PROBS))
#         GMM_PLOT = OUTPUT_DIR / f"gmm_plot_{args.name}.png"
#         plt.clf()
#         sns.kdeplot(x=FUCCI_intensities[:, 0], y=FUCCI_intensities[:, 1], hue=probs.argmax(dim=1), palette="Set2")
#         plt.savefig(GMM_PLOT)
#         plt.clf()
#         print("Saved GMM plot to " + str(GMM_PLOT))

if args.labels or args.all:
    print("Calculating angular, pseudotime, and phase class labels for the samples")
    assert not no_name, "Name of dataset must be specified"
    assert NAME_INDEX.exists(), "Index file for single cell image dataset does not exist, run --single_cell first"

    if ANGULAR_LABELS.exists() and PSEUDOTIME_LABELS.exists() and PHASE_LABELS.exists() and not args.rebuild:
        print("Labels files already exist, skipping. Set --rebuild to overwrite.")
    else:
        from multiprocessing import Pool
        from HPA_CC.data.img_stats import well_fucci_stats
        from HPA_CC.utils.pseudotime import intensities_to_pseudotime
        from sklearn.mixture import BayesianGaussianMixture

        image_paths, cell_mask_paths, nuclei_mask_paths = load_index_paths(NAME_INDEX)
        paths = list(zip(image_paths, nuclei_mask_paths))
        print(paths)
        with Pool(16) as pool:
            results = list(tqdm(pool.map(well_fucci_stats, paths), total=len(paths), desc="Getting FUCCI Statistics"))
        scope_names, cells_per_well, well_intensities, well_std_int, well_pseudotimes, well_angles = zip(*results)
        std_full_time, std_full_angles, std_full_std_int = intensities_to_pseudotime(np.concatenate(well_std_int), rescale=False, auto_start=True)

        NUM_COMPONENTS, NUM_CLASSES = 7, 4

        # Fit the Bayesian GMM to the angular distributions
        if args.scope:
            scope_names_list = []
            for scope, n_cell in zip(scope_names, cells_per_well):
                scope_names_list.extend([scope] * n_cell)
            # these are set in the pseudotime_labels.ipynb notebook at the botoom
            scope_gmms = pkl.load(open(OUTPUT_DIR / "scope_gmms.pkl", "rb"))
            groupings = pkl.load(open(OUTPUT_DIR / "groupings.pkl", "rb"))
            from HPA_CC.utils.pseudotime import comp_to_class

            class_likelihoods = []
            for i, (scope, angle) in tqdm(enumerate(zip(scope_names_list, std_full_angles)), desc="Calculating class likelihoods"):
                gmm = scope_gmms[scope]
                group = groupings[scope]
                sample_class_likelihoods = np.zeros((1, NUM_CLASSES))
                angles = np.array([angle])
                likelihoods = gmm._estimate_weighted_log_prob(angles.reshape(-1, 1))
                for comp in range(gmm.n_components):
                    cell_idx = sample_class_likelihoods[:, comp_to_class(comp, gmm, group)]
                    sample_class_likelihoods[:, comp_to_class(comp, gmm, group)] = np.log(np.exp(cell_idx) + np.exp(likelihoods[:, comp]))
                class_likelihoods.append(sample_class_likelihoods)
            class_likelihoods = np.concatenate(class_likelihoods)
            print(class_likelihoods.shape)
        else:
            gmm = BayesianGaussianMixture(n_components=7, covariance_type='full', n_init=10)
            gmm.fit(std_full_angles.reshape(-1, 1))

            def full_comp_to_class(comp):
                sorted_idx = list(np.argsort(gmm.means_.flatten()))
                if sorted_idx.index(comp) == 0:
                    return 0
                return ((sorted_idx.index(comp) - 1) // 2) + 1

            class_likelihoods = np.zeros((len(std_full_std_int), NUM_CLASSES))
            likelihoods = gmm._estimate_weighted_log_prob(std_full_angles.reshape(-1, 1))
            for comp in range(gmm.n_components):
                cell_idx = class_likelihoods[:, full_comp_to_class(comp)]
                class_likelihoods[:, full_comp_to_class(comp)] = np.log(np.exp(cell_idx) + np.exp(likelihoods[:, comp]))
        
        pseudotime_class = np.argmax(class_likelihoods, axis=1)

        torch.save(torch.tensor(std_full_angles), ANGULAR_LABELS) # [0, 1]
        torch.save(torch.tensor(std_full_time), PSEUDOTIME_LABELS) # [0, 1]
        torch.save(torch.tensor(class_likelihoods), PHASE_LABELS) # shape (n_samples, 4), likelihoods of each class
        
        n_cols = 12
        n_wells = len(well_angles)
        n_rows = n_wells // n_cols + 1
        color_map = {0: 'darkred', 1: 'red', 2: 'orange', 3: 'green'}
        plt.clf()
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True, sharey=True)
        cell_idx = 0
        for i, (well_path, n_cells, well_angle, intensities) in enumerate(zip(paths, cells_per_well, well_angles, well_std_int)):
            well = well_path[0].parent.name
            ps_class = pseudotime_class[cell_idx:cell_idx + n_cells]
            cell_idx += n_cells
            colors = [color_map[c] for c in ps_class]
            ax = axs[i // n_cols, i % n_cols]
            ax.set_title(f"{well} n={n_cells}")
            ax.scatter(intensities[:, 0], intensities[:, 1], c=colors)
        plt.savefig(OUTPUT_DIR / f"well_pseudotime_labels_{args.name}.png")
        plt.close()
        