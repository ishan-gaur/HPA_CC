import os
import csv
import subprocess
from pathlib import Path 
from glob import glob
from packaging import version
import warnings
from tqdm import tqdm

import torch
import numpy as np

import cv2
from scipy import ndimage
from microfilm.microplot import microshow
from skimage import measure, segmentation, morphology
from HPA_CC.utils.img_norm import min_max_normalization, percentile_normalization, image_cells_sharpness
import HPA_CC.utils.img_norm as img_norm

silent = False
suppress_warnings = False

def run_silent():
    global silent
    silent = True
    img_norm.silent = True

def has_channel_names(data_dir):
    return os.path.exists(data_dir / "channel_names.txt")

def save_channel_names(data_dir, channel_names):
    with open(data_dir / "channel_names.txt", "w+") as f:
        f.write("\n".join(channel_names))

def load_channel_names(data_dir):
    if not has_channel_names(data_dir):
        raise Exception(f"Channel names not found at: {data_dir}. Please check that you've provided them to the CLI or main function.")
    with open(data_dir / "channel_names.txt", "r") as f:
        channel_names = f.read().splitlines()
    return channel_names

def create_image_paths_file(data_dir, level, exists_ok=True, overwrite=False):
    if type(data_dir) == Path:
        data_dir = str(data_dir)
    data_paths_file = data_dir / "data-folder.txt"

    if data_paths_file.exists():
        if not exists_ok:
            raise Exception(f"Image path index already exists at: {data_paths_file}")
        if overwrite:
            print(f"Overwriting image path index at: {data_paths_file}")
            os.remove(data_paths_file)
        else:
            print(f"Image path index found at: {data_paths_file}")

    if not os.path.exists(data_paths_file) or overwrite:
        print(f"Creating image path index at: {data_paths_file}")
        if level == -1:
            # find max depth of the directory
            print("WARNING: depth -1 is not tested, use at your own risk")
            p = subprocess.run(f"find {data_dir} -type d | awk -F/ '{{print NF-1}}' | sort -nu | tail -n 1", shell=True, capture_output=True)
            level = int(p.stdout.decode("utf-8").strip())
        p = subprocess.run(f"find {data_dir} -mindepth {level} -maxdepth {level} -type d", shell=True, capture_output=True)
        with open(data_paths_file, "w+") as f:
            f.write(p.stdout.decode("utf-8"))
    
    p = subprocess.run(f"cat {data_paths_file} | wc -l", shell=True, capture_output=True)
    num_paths = int(p.stdout.decode("utf-8").strip())
    print(f"Number of target paths found: {num_paths}")
    return data_paths_file, num_paths

def image_paths_from_folders(folders_file):
    image_paths = list(open(folders_file, "r"))
    image_paths = [Path(x.strip()) for x in image_paths]
    return image_paths

def segmentator_setup(multi_channel_model, device):
    if version.parse(torch.__version__) >= version.parse("1.10.0"):
        raise ValueError(f"HPA Cell Segmentator is not compatible with torch >= 1.10.0.\nTorch {torch.__version__} detected. Are you using the 'data-prep' conda environment?")
    if version.parse(np.__version__) >= version.parse("1.20.0"):
        raise ValueError(f"HPA Cell Segmentator is not compatible with torch >= 1.10.0.\nTorch {torch.__version__} detected. Are you using the 'data-prep' conda environment?")
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _segmentator_setup(multi_channel_model, device)
    else:
        return _segmentator_setup(multi_channel_model, device)

def _segmentator_setup(multi_channel_model, device):
    import hpacellseg.cellsegmentator as cellsegmentator
    pwd = Path(os.getcwd())
    NUC_MODEL = pwd / "HPA-Cell-Segmentation" / "nuclei-model.pth"
    CELL_MODEL = pwd / "HPA-Cell-Segmentation" / "cell-model.pth"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    segmentator = cellsegmentator.CellSegmentator(
        str(NUC_MODEL), str(CELL_MODEL), device=device, padding=True, multi_channel_model=multi_channel_model
    )
    return segmentator

def segmentations_exist(image_path):
    if (image_path / "images.npy").exists():
        if (image_path / "cell_masks.npy").exists():
            if (image_path / "nuclei_masks.npy").exists():
                return True
    return False

def clean_segmentations(image_paths): 
    for image_path in image_paths:
            if (image_path / "images.npy").exists():
                os.remove(image_path / "images.npy") 
            if (image_path / "cell_masks.npy").exists():
                os.remove(image_path / "cell_masks.npy")
            if (image_path / "nuclei_masks.npy").exists():
                os.remove(image_path / "nuclei_masks.npy")

def merge_extraneous_elements(targets, elements):
    extraneous_elements = np.unique(elements[~np.isin(elements, targets)])
    extraneous_elements = extraneous_elements[extraneous_elements != 0] # remove background
    target_masks = np.stack([targets == target for target in np.unique(targets)])
    target_coms = np.array([ndimage.center_of_mass(mask) for mask in target_masks])
    for element in extraneous_elements:
        element_com = ndimage.center_of_mass(elements == element)
        target_distances = np.linalg.norm(target_coms - element_com, axis=1)
        closest_target = np.unique(targets)[np.argmin(target_distances)]
        elements[elements == element] = closest_target
    return elements

def need_rebuild(file, rebuild):
    if not file.exists():
        return True
    elif rebuild:
        os.remove(file)
        return True
    else:
        return False


def glob_channel_images(image_path, channel):
    return list(glob(f"{str(image_path)}/**/*{channel}.png", recursive=True))

def load_image_from_file(path_list):
    return [cv2.imread(str(x), cv2.IMREAD_UNCHANGED) for x in path_list]

def composite_images_from_paths(image_paths, channel_names):
    path_images = []
    for image_path in tqdm(image_paths, desc="Loading images"):
        channels = []
        for c in channel_names:
            channel_paths = sorted(glob_channel_images(image_path, c))
            channel_images = load_image_from_file(channel_paths)
            channel_images = np.stack(channel_images, axis=0)
            channels.append(channel_images)
        path_images.append(np.stack(channels, axis=1))
    path_images = np.concatenate(path_images)
    return path_images

def get_masks(segmentator, image_paths, channel_names, dapi, tubl, calb2, merge_missing=True, rebuild=False, display=False):
    if version.parse(torch.__version__) >= version.parse("1.10.0"):
        raise ValueError(f"HPA Cell Segmentator is not compatible with torch >= 1.10.0.\nTorch {torch.__version__} detected. Are you using the 'data-prep' conda environment?")
    if version.parse(np.__version__) >= version.parse("1.20.0"):
        raise ValueError(f"HPA Cell Segmentator is not compatible with torch >= 1.10.0.\nTorch {torch.__version__} detected. Are you using the 'data-prep' conda environment?")
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _get_masks(segmentator, image_paths, channel_names, dapi, tubl, calb2, merge_missing, rebuild, display)
    else:
        return _get_masks(segmentator, image_paths, channel_names, dapi, tubl, calb2, merge_missing, rebuild, display)

def _get_masks(segmentator, image_paths, channel_names, dapi, tubl, calb2, merge_missing=True, rebuild=False, display=False):
    """
    Get the masks for the images in image_paths using HPA-Cell-Segmentation
    """
    from hpacellseg.utils import label_cell
    images_paths = []
    nuclei_mask_paths = []
    cell_mask_paths = []
    for image_path in tqdm(image_paths, desc="Getting masks"):
        if ((not need_rebuild(image_path / "images.npy", rebuild)) and
            (not need_rebuild(image_path / "cell_masks.npy", rebuild)) and
            (not need_rebuild(image_path / "nuclei_masks.npy", rebuild))):
            images_paths.append(image_path / "images.npy")
            nuclei_mask_paths.append(image_path / "nuclei_masks.npy")
            cell_mask_paths.append(image_path / "cell_masks.npy")
            continue

        dapi_paths = sorted(glob_channel_images(image_path, channel_names[dapi]))
        tubl_paths = sorted(glob_channel_images(image_path, channel_names[tubl]))
        calb2_paths = sorted(glob_channel_images(image_path, channel_names[calb2])) if calb2 is not None else None

        if len(dapi_paths) == 0 or len(tubl_paths) == 0:
            print(f"Missing DAPI or TUBULIN image in {image_path}")
            print("\t", os.listdir(image_path))
            continue
        
        for dapi_path, tubl_path in zip(dapi_paths, tubl_paths):
            assert str(dapi_path).split(channel_names[dapi])[0] == str(tubl_path).split(channel_names[tubl])[0], f"File mismatch for {dapi_path} and {tubl_path}"
        if calb2 is not None and calb2_paths is not None:
            for dapi_path, anln_path in zip(dapi_paths, calb2_paths):
                assert str(dapi_path).split(channel_names[dapi])[0] == str(anln_path).split(channel_names[calb2])[0], f"File mismatch for {dapi_path} and {anln_path}"

        
        dapi_images = load_image_from_file(dapi_paths)
        tubl_images = load_image_from_file(tubl_paths)
        calb2_images = load_image_from_file(calb2_paths) if calb2_paths is not None else None

        ref_images = [tubl_images, calb2_images, dapi_images]
        nuc_segmentation = segmentator.pred_nuclei(ref_images[2])
        cell_segmentation = segmentator.pred_cells(ref_images)

        # post-processing
        nuclei_masks, cell_masks = [], []
        for i in range(len(ref_images[2])): # 2 because DAPI will always be present and we set the order manually when defining ref_images
            nuclei_mask, cell_mask = label_cell(
                nuc_segmentation[i], cell_segmentation[i]
            )
            nuclei_masks.append(nuclei_mask)
            cell_masks.append(cell_mask)
        nuclei_masks = np.stack(nuclei_masks, axis=0)
        cell_masks = np.stack(cell_masks, axis=0)

        # apply preprocessing mask if the user want to merge nuclei
        images = []
        for c, channel in enumerate(channel_names):
            channel_paths = sorted(glob_channel_images(image_path, channel_names[c]))
            channel_images = load_image_from_file(channel_paths)
            channel_images = np.stack(channel_images, axis=0)
            images.append(channel_images)
        images = np.stack(images, axis=1)

        for i in [0, -2, -1]:
            assert images.shape[i] == nuclei_masks.shape[i] == cell_masks.shape[i], f"Shape mismatch for images and masks in {image_path}, at index {i}, images has shape {images.shape}, nuclei_masks has shape {nuclei_masks.shape}, cell_masks has shape {cell_masks.shape}"

        image_idx = 0
        for image, nuclei_mask, cell_mask in zip(images, nuclei_masks, cell_masks):
            if set(np.unique(nuclei_mask)) != set(np.unique(cell_mask)): 
                print(f"Mask mismatch for {image_path}, nuclei: {np.unique(nuclei_masks)}, cell: {np.unique(cell_masks)}")
                if display:
                    microshow(image[(dapi, tubl),], cmaps=["pure_blue", "pure_red"], label_text=f"Image: {Path(image_path).name}[{image_idx}] ")

                # show cells without nuclei if any
                missing_nuclei = np.asarray(list(set(np.unique(cell_mask)) - set(np.unique(nuclei_mask))))
                if len(missing_nuclei) > 0 and display:
                    microshow(image[tubl] * np.isin(cell_mask, missing_nuclei), cmaps=["pure_red"], label_text=f"Cells missing nuclei: {missing_nuclei}")

                # show nuclei without cells if any
                missing_cells = np.asarray(list(set(np.unique(nuclei_mask)) - set(np.unique(cell_mask))))
                if len(missing_cells) > 0 and display:
                    microshow(image[dapi] * np.isin(nuclei_mask, missing_cells), cmaps=["pure_blue"], label_text=f"Nuclei without cells: {missing_cells}")

                # show cell masks and merge missing cells based on nuclei
                if display:
                    microshow(cell_mask, label_text=f"Cell mask: {np.unique(cell_mask)}")
                if len(missing_nuclei) > 0 and merge_missing:
                    cell_mask = merge_extraneous_elements(nuclei_mask, cell_mask)
                    if display:
                        microshow(cell_mask, label_text=f"Cell mask after merging: {np.unique(cell_mask)}")

                # show nuclei masks and merge missing nuclei based on cells
                microshow(nuclei_mask, label_text=f"Nuclei mask: {np.unique(nuclei_mask)}")
                if len(missing_cells) > 0 and merge_missing:
                    nuclei_mask = merge_extraneous_elements(cell_mask, nuclei_mask)
                    if display:
                        microshow(nuclei_mask, label_text=f"Nuclei mask after merging: {np.unique(nuclei_mask)}")

        for i, (nuclei_mask, cell_mask) in enumerate(zip(nuclei_masks, cell_masks)):
            assert np.max(nuclei_mask) > 0 and np.max(cell_mask) > 0, f"No nuclei or cell mask found for {image_path}"
            assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch for {image_path}, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"

        np.save(image_path / "images.npy", images)
        np.save(image_path / "nuclei_masks.npy", nuclei_masks)
        np.save(image_path / "cell_masks.npy", cell_masks)

        images_paths.append(image_path / "images.npy")
        nuclei_mask_paths.append(image_path / "nuclei_masks.npy")
        cell_mask_paths.append(image_path / "cell_masks.npy")

    return images_paths, nuclei_mask_paths, cell_mask_paths

def clear_border(cell_mask, nuclei_mask):
    # inside clear_border they make borders by dimension. So if you have a 2D image, 
    # they make a 2D border, if you have a 3D image, they make a 3D border, which means the
    # first and last channels are fully borders. So we need to squeeze the image to 2D beforehand
    cell_mask, nuclei_mask = np.squeeze(cell_mask), np.squeeze(nuclei_mask)
    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch before clearing border, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"

    num_removed = 0
    cleared_nuclei_mask = segmentation.clear_border(nuclei_mask)
    kept_values = np.unique(cleared_nuclei_mask)
    kept_cells = np.isin(cell_mask, kept_values)
    cleared_cell_mask = cell_mask * kept_cells
    num_removed = len(np.unique(nuclei_mask)) - len(kept_values)
    if num_removed == np.max(nuclei_mask):
        assert np.max(kept_values) == 0, f"Something went wrong with clearing the border, num_removed is the same as the highest index mask in nuclei mask, but the keep_value {np.max(kept_values)} != 0"
    nuclei_mask = cleared_nuclei_mask
    cell_mask = cleared_cell_mask

    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch after clearing border, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"
    return cell_mask, nuclei_mask, num_removed

def remove_small_objects(cell_mask, nuclei_mask, min_size=1000):
    num_removed = 0

    cell_mask = morphology.remove_small_objects(cell_mask, min_size=min_size)
    num_removed = len(np.unique(nuclei_mask)) - len(np.unique(cell_mask))
    nuclei_mask = nuclei_mask * (cell_mask > 0)
    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch after removing small objects, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"

    nuclei_mask = morphology.remove_small_objects(nuclei_mask, min_size=min_size)
    num_removed += len(np.unique(cell_mask)) - len(np.unique(nuclei_mask))
    cell_mask = cell_mask * np.isin(cell_mask, np.unique(nuclei_mask))
    assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch after removing small objects, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"

    return cell_mask, nuclei_mask, num_removed

def merge_nuclei(nuclei_mask, cell_mask, dialation_radius=20):
    bin_nuc_mask = (nuclei_mask > 0).astype(np.int8)
    cls_nuc = morphology.closing(bin_nuc_mask, morphology.disk(dialation_radius))
    # get the labels of touching nuclei
    new_label_map = morphology.label(cls_nuc)
    new_label_idx = np.unique(new_label_map)[1:]

    new_cell_mask = np.zeros_like(cell_mask)
    new_nuc_mask = np.zeros_like(nuclei_mask)
    for new_label in new_label_idx:
        # get the label of the touching nuclei
        old_labels = np.unique(nuclei_mask[new_label_map == new_label])
        old_labels = old_labels[old_labels != 0]

        new_nuc_mask[np.isin(nuclei_mask, old_labels)] = new_label
        new_cell_mask[np.isin(cell_mask, old_labels)] = new_label

        # for old_label in old_labels:
        #     new_cell_mask[cell_mask == old_label] = new_label
        #     new_nuc_mask[nuclei_mask == old_label] = new_label
    return new_nuc_mask, new_cell_mask


def clean_cell_masks(
    cell_mask,
    nuclei_mask,
    remove_size=0, # remove cells smaller than remove_size, based on the area of the bounding box, honestly could be higher, mb 2500. Make 0 to turn off.
    dialation_radius=0, # this is for 2048x2048 images adjust as needed. Make 0 to turn off.
):
    num_removed = 0
    ### see if nuclei are touching and merge them
    if dialation_radius > 0:
        nuclei_mask, cell_mask = merge_nuclei(nuclei_mask, cell_mask, dialation_radius=dialation_radius)
        assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask))
    else:
        cell_mask = cell_mask
        nuclei_mask = nuclei_mask

    region_props = measure.regionprops(cell_mask, (cell_mask > 0).astype(np.uint8))
    pre_size = len(region_props)
    if remove_size > 0:
        region_props = [x for x in region_props if x.area > remove_size]
        num_removed += pre_size - len(region_props)
    bbox_array = np.array([x.bbox for x in region_props])

    # convert x1,y1,x2,y2 to x,y,w,h
    bbox_array[:, 2] = bbox_array[:, 2] - bbox_array[:, 0]
    bbox_array[:, 3] = bbox_array[:, 3] - bbox_array[:, 1]

    # com_array = np.array([x.weighted_centroid for x in region_props])

    return cell_mask, nuclei_mask, bbox_array, num_removed

def relabel_masks(cell_mask, nuclei_mask):
    # relabel the masks so that the labels are consecutive
    # this is necessary because the masks may have been modified
    # and may have missing labels
    new_nuclei_mask = np.zeros_like(nuclei_mask)
    new_cell_mask = np.zeros_like(cell_mask)
    for i, label in enumerate(np.unique(nuclei_mask)):
        new_nuclei_mask[nuclei_mask == label] = i
        new_cell_mask[cell_mask == label] = i
    return new_cell_mask, new_nuclei_mask

def clean_and_save_masks(
    cell_mask_paths,
    nuclei_mask_paths,
    clean_suffix,
    rm_border=True, # removes cells with nuclei touching the border
    remove_size=2500, # remove cells smaller than remove_size, based on the area of the bounding box, honestly could be higher, mb 2500. Make 0 to turn off.
    # dialation_radius=0, # this is for 2048x2048 images adjust as needed. Make 0 to turn off.
):
    num_original = 0
    num_removed = 0
    new_cell_paths, new_nuclei_paths = [], []
    for (cell_mask_path, nuclei_mask_path) in tqdm(list(zip(cell_mask_paths, nuclei_mask_paths)), desc="Cleaning masks"):
        cell_mask = np.load(cell_mask_path)
        nuclei_mask = np.load(nuclei_mask_path)
        if cell_mask.ndim == 2:
            cell_mask = np.expand_dims(cell_mask, axis=0)
        if nuclei_mask.ndim == 2:
            nuclei_mask = np.expand_dims(nuclei_mask, axis=0)
        assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch before cleaning, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"
        for i in range(len(nuclei_mask)):
            num_original += len(np.unique(nuclei_mask[i]))
        if rm_border:
            for i in range(len(cell_mask)):
                num_start = np.array([len(np.unique(cell_mask[i])), len(np.unique(nuclei_mask[i]))])
                cell_mask[i], nuclei_mask[i], n_removed = clear_border(cell_mask[i], nuclei_mask[i])
                num_removed += n_removed
                num_end = np.array([len(np.unique(cell_mask[i])), len(np.unique(nuclei_mask[i]))])
                assert np.all(num_start - num_end >= n_removed), f"Something went wrong with clearing the border, num_start {num_start} - num_end {num_end} < 0"
        if remove_size > 0:
            cell_mask, nuclei_mask, n_removed = remove_small_objects(cell_mask, nuclei_mask, min_size=remove_size)
            num_removed += n_removed
        assert set(np.unique(nuclei_mask)) == set(np.unique(cell_mask)), f"Mask mismatch after cleaning, nuclei: {np.unique(nuclei_mask)}, cell: {np.unique(cell_mask)}"
        cell_mask, nuclei_mask = relabel_masks(cell_mask, nuclei_mask)
        new_cell_path = cell_mask_path.parent / f"{cell_mask_path.stem}_{clean_suffix}"
        new_nuclei_path = nuclei_mask_path.parent / f"{nuclei_mask_path.stem}_{clean_suffix}"
        np.save(new_cell_path, cell_mask)
        np.save(new_nuclei_path, nuclei_mask)
        new_cell_paths.append(new_cell_path.parent / (new_cell_path.stem + ".npy"))
        new_nuclei_paths.append(new_nuclei_path.parent / (new_nuclei_path.stem + ".npy"))
    return new_cell_paths, new_nuclei_paths, num_original, num_removed


def crop_images(image_paths, cell_mask_paths, nuclei_mask_paths, crop_size, nuc_margin=50):
    # images need to be C x H x W

    # CACHE FILE LEVEL ITERATION
    # print(f"Found {len(image_paths)} cache files")
    seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths = [], [], []
    total_cells = 0
    for (image_path, cell_mask_path, nuclei_mask_path) in tqdm(list(zip(image_paths, cell_mask_paths, nuclei_mask_paths)), desc="Cropping images"):
        cache_images = np.load(image_path)
        cache_cell_masks = np.load(cell_mask_path)
        cache_nuclei_masks = np.load(nuclei_mask_path)
        if cache_images.ndim == 3:
            cache_images = np.expand_dims(cache_images, axis=0)
        if cache_cell_masks.ndim == 2:
            cache_cell_masks = np.expand_dims(cache_cell_masks, axis=0)
        if cache_nuclei_masks.ndim == 2:
            cache_nuclei_masks = np.expand_dims(cache_nuclei_masks, axis=0)

        assert (cache_cell_masks.astype(int).astype(cache_cell_masks.dtype) == cache_cell_masks).all(), f"Cell masks are not integers, {cell_masks.dtype}"
        assert (cache_nuclei_masks.astype(int).astype(cache_nuclei_masks.dtype) == cache_nuclei_masks).all(), f"Nuclei masks are not integers, {nuclei_masks.dtype}"

        cache_cell_masks, cache_nuclei_masks = cache_cell_masks.astype(int), cache_nuclei_masks.astype(int)

        # IMAGE LEVEL ITERATION
        # print(f"Found {len(cache_images)} images in {image_path}")
        seg_images, seg_cell_masks, seg_nuclei_masks = [], [], []
        for image, cell_masks, nuclei_masks in zip(cache_images, cache_cell_masks, cache_nuclei_masks):
            region_props = measure.regionprops(cell_masks)
            if len(region_props) == 0:
                print(f"No cells found in one of the {image_path} masks")
                continue
            nuclear_regions = measure.regionprops(nuclei_masks)
            bboxes = np.array([x.bbox for x in region_props])
            bbox_widths = bboxes[:, 2] - bboxes[:, 0]
            bbox_heights = bboxes[:, 3] - bboxes[:, 1]
            nuc_bboxes = np.array([x.bbox for x in nuclear_regions])

            # SINGLE CELL LEVEL ITERATION
            # print(f"Found {len(bboxes)} cells in {image_path}")
            cell_mask_list = []
            nuclei_mask_list = []
            image_list = []
            for i, (bbox, width, height, nbox) in enumerate(zip(bboxes, bbox_widths, bbox_heights, nuc_bboxes)):
                center = np.array([np.mean([bbox[0], bbox[2]]), np.mean([bbox[1], bbox[3]])])
                if width > crop_size:
                    left_edge = int(center[0] - crop_size / 2)
                    right_edge = int(center[0] + crop_size / 2)
                    if nbox[0] < left_edge and nbox[2] > right_edge:
                        print(f"nucleus {i} is wider than crop region, bbox: {bbox}, nbox: {nbox}, center: {center}, crop_size: {crop_size}")
                        continue
                    if nbox[0] < left_edge:
                        new_left = nbox[0] - nuc_margin
                        displacement = new_left - left_edge
                        left_edge = new_left
                        right_edge = right_edge + displacement
                    elif nbox[2] > right_edge:
                        new_right = nbox[2] + nuc_margin
                        displacement = new_right - right_edge
                        right_edge = new_right
                        left_edge = left_edge + displacement
                    slice_x = slice(max(0, left_edge), min(cell_masks.shape[0], right_edge)) 
                    padding_x = (max(0 - left_edge, 0), max(right_edge - cell_masks.shape[0], 0))
                else:
                    slice_x = slice(bbox[0], bbox[2])
                    padding_x = crop_size - width
                    padding_x = (padding_x // 2, padding_x - padding_x // 2)
                if height > crop_size:
                    top_edge = int(center[1] - crop_size / 2)
                    bottom_edge = int(center[1] + crop_size / 2)
                    if nbox[1] < top_edge and nbox[3] > bottom_edge:
                        print(f"nucleus {i} is taller than crop region, bbox: {bbox}, nbox: {nbox}, center: {center}, crop_size: {crop_size}")
                        continue
                    if nbox[1] < top_edge:
                        new_top = nbox[1] - nuc_margin
                        displacement = top_edge - new_top
                        top_edge = new_top
                        bottom_edge = bottom_edge - displacement
                    elif nbox[3] > bottom_edge:
                        new_bottom = nbox[3] + nuc_margin
                        displacement = new_bottom - bottom_edge
                        bottom_edge = new_bottom
                        top_edge = top_edge + displacement
                    slice_y = slice(max(0, top_edge), min(cell_masks.shape[1], bottom_edge))
                    padding_y = (max(0 - top_edge, 0), max(bottom_edge - cell_masks.shape[1], 0))
                else:
                    slice_y = slice(bbox[1], bbox[3])
                    padding_y = crop_size - height
                    padding_y = (padding_y // 2, padding_y - padding_y // 2)

                cell_mask = ((cell_masks[slice_x, slice_y] * np.isin(cell_masks[slice_x, slice_y], i + 1)) > 0)
                nuclei_mask = ((nuclei_masks[slice_x, slice_y] * np.isin(nuclei_masks[slice_x, slice_y], i + 1)) > 0)
                cell_image = image[:, slice_x, slice_y] * cell_mask
                
                cell_mask = np.pad(cell_mask, (padding_x, padding_y), mode="constant", constant_values=0)
                nuclei_mask = np.pad(nuclei_mask, (padding_x, padding_y), mode="constant", constant_values=0)
                cell_image = np.pad(cell_image, ((0, 0), padding_x, padding_y), mode="constant", constant_values=0)

                cell_mask_list.append(cell_mask)
                nuclei_mask_list.append(nuclei_mask)
                image_list.append(cell_image)

            cell_masks = np.stack(cell_mask_list, axis=0)
            nuclei_masks = np.stack(nuclei_mask_list, axis=0)
            images = np.stack(image_list, axis=0)

            seg_images.append(images)
            seg_cell_masks.append(cell_masks)
            seg_nuclei_masks.append(nuclei_masks)

        if len(seg_images) == 0:
            print(f"No cells found in {image_path}")
            continue

        total_cells += len(seg_images)
        # print(f"Found {len(seg_images)} cells in {image_path}")

        seg_images = np.concatenate(seg_images, axis=0)
        seg_cell_masks = np.concatenate(seg_cell_masks, axis=0)
        seg_nuclei_masks = np.concatenate(seg_nuclei_masks, axis=0)

        # print(seg_images.shape)

        seg_cell_mask_paths.append(cell_mask_path.parent /"seg_cell_masks.npy")
        seg_nuclei_mask_paths.append(nuclei_mask_path.parent / "seg_nuclei_masks.npy")
        seg_image_paths.append(image_path.parent / "seg_images.npy")

        np.save(seg_cell_mask_paths[-1], seg_cell_masks)
        np.save(seg_nuclei_mask_paths[-1], seg_nuclei_masks)
        np.save(seg_image_paths[-1], seg_images)

    # print(f"Total cells: {total_cells}")

    return seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths

channel_min = lambda x: torch.min(torch.min(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
channel_max = lambda x: torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]

def resize(seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths, target_dim, name_suffix, resize_type=cv2.INTER_LANCZOS4):
    target_dim = (target_dim, target_dim)
    final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths = [], [], []
    for (seg_image_path, seg_cell_mask_path, seg_nuclei_mask_path) in tqdm(list(zip(seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths)), desc="Resizing images"):
        images = np.load(seg_image_path) # B x C x H x W
        cell_masks = np.load(seg_cell_mask_path) # B x H x W
        nuclei_masks = np.load(seg_nuclei_mask_path) # B x H x W

        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        if cell_masks.ndim == 2:
            cell_masks = np.expand_dims(cell_masks, axis=0)
        if nuclei_masks.ndim == 2:
            nuclei_masks = np.expand_dims(nuclei_masks, axis=0)
        
        images = np.transpose(images, (0, 2, 3, 1)) # B x H x W x C

        resized_images, resized_cell_masks, resized_nuclei_masks = [], [], []
        for image, cell_mask, nuclei_mask in zip(images, cell_masks, nuclei_masks):
            resized_images.append(cv2.resize(image, dsize=target_dim, interpolation=resize_type))
            assert np.max(np.unique(cell_mask)) <= 1, f"Cell mask has more than 1 unique value, {np.unique(cell_mask)}"
            resized_cell_masks.append(cv2.resize(cell_mask.astype("float32"), dsize=target_dim, interpolation=cv2.INTER_NEAREST))
            assert np.max(np.unique(nuclei_mask)) <= 1, f"Nuclei mask has more than 1 unique value, {np.unique(nuclei_mask)}"
            resized_nuclei_masks.append(cv2.resize(nuclei_mask.astype("float32"), dsize=target_dim, interpolation=cv2.INTER_NEAREST))

        resized_images = np.stack(resized_images, axis=0)
        resized_images = np.transpose(resized_images, (0, 3, 1, 2)) # B x C x H x W
        resized_images = torch.Tensor(resized_images.astype("float32"))

        IMAGE, CELL_MASK, NUCLEI_MASK = f"images_{name_suffix}.pt", f"cell_masks_{name_suffix}.pt", f"nuclei_masks_{name_suffix}.pt"
        torch.save(resized_images, seg_image_path.parent / IMAGE)
        final_image_paths.append(seg_image_path.parent / IMAGE)
        torch.save(torch.Tensor(np.array(resized_cell_masks)), seg_cell_mask_path.parent / CELL_MASK)
        final_cell_mask_paths.append(seg_cell_mask_path.parent / CELL_MASK)
        torch.save(torch.Tensor(np.array(resized_nuclei_masks)), seg_nuclei_mask_path.parent / NUCLEI_MASK)
        final_nuclei_mask_paths.append(seg_nuclei_mask_path.parent / NUCLEI_MASK)

    return final_image_paths, final_cell_mask_paths, final_nuclei_mask_paths

def create_data_path_index(image_paths, cell_mask_paths, nuclei_mask_paths, index_file, overwrite=False):
    if os.path.exists(index_file) and not overwrite:
        print("Index file already exists, not overwriting")
        return

    sample_paths = []
    for image_path, cell_mask_path, nuclei_mask_path in zip(image_paths, cell_mask_paths, nuclei_mask_paths):
        sample_paths.append({
            "sample_name": image_path.parent,
            "image_path": str(image_path),
            "cell_mask_path": str(cell_mask_path),
            "nuclei_mask_path": str(nuclei_mask_path)
        })

    with open(index_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_name", "image_path", "cell_mask_path", "nuclei_mask_path"])
        writer.writeheader()
        for sample_path in sample_paths:
            writer.writerow(sample_path)

def load_data_path_index(data_dir, file_name="index.csv"):
    index_file = data_dir / file_name
    if not index_file.exists():
        raise ValueError(f"Index file {index_file} does not exist")
    with open(index_file, "r") as f:
        reader = csv.DictReader(f)
        sample_paths = []
        for row in reader:
            sample_paths.append(row)
    return sample_paths

def load_index_paths(index_file):
    data_dir = Path(index_file).parent
    file_name = Path(index_file).name
    sample_paths = load_data_path_index(data_dir, file_name)
    image_paths, cell_mask_paths, nuclei_mask_paths = [], [], []
    for sample_path in sample_paths:
        image_paths.append(Path(sample_path["image_path"]))
        cell_mask_paths.append(Path(sample_path["cell_mask_path"]))
        nuclei_mask_paths.append(Path(sample_path["nuclei_mask_path"]))
    return image_paths, cell_mask_paths, nuclei_mask_paths

def load_dir_images(index_file, num_load=None):
    image_paths, _, _ = load_index_paths(index_file)
    if num_load is not None:
        image_paths = image_paths[:num_load]
    images = []
    for image_path in tqdm(image_paths, desc="Loading images"):
        images.append(torch.load(Path(image_path)))
    return images

def normalize_images(image_paths, cell_masks_paths, norm_strategy, norm_min, norm_max, norm_suffix, batch_size=100, save_samples=False, cmaps=None):
    # batch_size needs to be 1 if using well-level normalization
    # maybe I just need to make a normalize_wells function
    if save_samples and not cmaps:
        raise ValueError("Need to provide cmaps if saving samples")

    import data_viz
    data_viz.silent = True
    from data_viz import save_image, save_image_grid

    paths = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Normalizing images"):
        batch_paths = image_paths[i:min(i+batch_size, len(image_paths))]
        batch_mask_paths = cell_masks_paths[i:min(i+batch_size, len(image_paths))]
        images = []
        masks = []
        ranges = [0]
        for image_path, cell_path in zip(batch_paths, batch_mask_paths):
            batch_elem_images = np.load(image_path)
            ranges.append(ranges[-1] + len(batch_elem_images))
            images.append(batch_elem_images)
            masks.append(np.load(cell_path))
        images = np.concatenate(images, axis=0)
        masks = np.concatenate(masks, axis=0)[:, None, ...]
        images = images * (masks > 0)

        # print(f"Found {len(images)} images in batch {i}")

        if norm_strategy == "min_max":
            images = min_max_normalization(images, stats=False)
            # print(f"Found {len(images)} after norm")
        elif norm_strategy == "percentile":
            if norm_min is None or norm_max is None:
                raise ValueError("Must provide norm_min and norm_max for percentile normalization")
            images = percentile_normalization(images, norm_min, norm_max, stats=False)
        else:
            raise NotImplementedError(f"Normalization strategy {norm_strategy} not implemented in CLI")

        if save_samples:
            image = images[-1]
            old_image = np.load(batch_paths[-1])[-1].astype("float32")
            for c in range(image.shape[0]):
                old_image[c] = (old_image[c] - old_image[c].min()) / (old_image[c].max() - old_image[c].min())
            old_image = old_image * (masks[-1] > 0)
            path = batch_paths[-1]
            path = path.parent / (f"{path.stem}{norm_suffix}" + ".png")
            sample_images = torch.tensor(np.concatenate([old_image, image]))
            sample_images = sample_images[:, None, ...]
            save_image_grid(sample_images, path, nrow=image.shape[0], cmaps=None)

        # print(images.shape)
        for j, path in enumerate(batch_paths):
            if norm_suffix is not None:
                path = path.parent / (f"{path.stem}{norm_suffix}" + path.suffix)
            # print(f"Saving {j}: {ranges[j]} to {ranges[j+1]}")
            np.save(path, images[ranges[j]:ranges[j+1]])
            paths.append(path)
    return paths

# def filter_images_by_sharpness(seg_image_paths, threshold, sharp_suffix, seg_cell_mask_paths=None, seg_nuclei_mask_paths=None):
#     if seg_cell_mask_paths is None:
#         seg_cell_mask_paths = [None] * len(seg_image_paths)
#     if seg_nuclei_mask_paths is None:
#         seg_nuclei_mask_paths = [None] * len(seg_image_paths)
#     num_removed, num_total = 0, 0
#     for images_path, cell_mask_path, nuclei_mask_path in tqdm(zip(seg_image_paths, seg_cell_mask_paths, seg_nuclei_mask_paths), desc="Filtering by sharpness"):
#         images = torch.tensor(np.load(images_path).astype("float32").squeeze())
#         sharpness = sample_sharpness(images)
#         num_images = len(sharpness)
#         images = images[sharpness > threshold]
#         if cell_mask_path is not None:
#             cell_masks = torch.tensor(np.load(cell_mask_path).astype("float32").squeeze())
#             cell_masks = cell_masks[sharpness > threshold]
#             np.save(cell_mask_path, cell_masks.numpy())
#         if nuclei_mask_path is not None:
#             nuclei_masks = torch.tensor(np.load(nuclei_mask_path).astype("float32").squeeze())
#             nuclei_masks = nuclei_masks[sharpness > threshold]
#             np.save(nuclei_mask_path, nuclei_masks.numpy())
#         sharpness = sharpness[sharpness > threshold]
#         num_removed += num_images - len(sharpness)
#         num_total += num_images
#         np.save(images_path, images.numpy())
    # return num_removed, num_total

def filter_masks_by_sharpness(image_paths, cell_mask_paths, nuclei_mask_paths, threshold, dapi, tubl, sharp_suffix, save_samples=True, cmaps=None):
    if save_samples and not cmaps:
        raise ValueError("Need to provide cmaps if saving samples")
    import data_viz
    data_viz.silent = True
    from data_viz import save_image
    new_cell_paths, new_nuclei_paths = [], []
    num_removed, num_total = 0, 0
    for images_path, cell_mask_path, nuclei_mask_path in tqdm(zip(image_paths, cell_mask_paths, nuclei_mask_paths),
                                                              desc="Filtering by sharpness", total=len(image_paths)):
        images = np.load(images_path).astype("float32")
        cell_masks = np.load(cell_mask_path).astype("float32")
        nuclei_masks = np.load(nuclei_mask_path).astype("float32")
        print(f"Found {len(images)} images in {images_path}")
        for i in range(len(images)):
            image, cell_mask = images[i, (dapi, tubl)], cell_masks[i] # sharpness only on dapi and tubl to generalize better
            num_cells = len(np.unique(cell_mask)) - 1 # -1 because the background is 0
            sharpness_levels = image_cells_sharpness(torch.tensor(image), cell_mask)
            sharpness_levels = np.array([threshold if s is None else s for s in sharpness_levels])
            keep_set = np.where(sharpness_levels > threshold)[0]
            cell_mask = cell_mask * np.isin(cell_mask, keep_set)
            nuclei_mask = nuclei_masks[i] * np.isin(cell_mask, keep_set)
            cell_masks[i], nuclei_masks[i] = relabel_masks(cell_mask, nuclei_mask)
            num_total += num_cells
            num_removed += num_cells - (len(np.unique(cell_mask)) - 1) # -1 because the background is 0
        new_cell_path = cell_mask_path.parent / f"{cell_mask_path.stem}_sharp_{sharp_suffix}"
        new_nuclei_path = nuclei_mask_path.parent / f"{nuclei_mask_path.stem}_sharp_{sharp_suffix}"
        if save_samples:
            old_mask = np.load(cell_mask_path).astype("float32")[-1]
            new_mask = cell_masks[-1]
            cmaps = [cmaps[dapi], cmaps[tubl]]
            save_image(torch.tensor(image * ((new_mask == 0) & (old_mask != 0))), new_cell_path.parent / f"sharp_removed_{new_cell_path.stem}.png", cmaps)
        np.save(new_cell_path, cell_masks)
        np.save(new_nuclei_path, nuclei_masks)
        print(f"{sum([len(np.unique(x)) - 1 for x in cell_masks])} cells left after filtering for {images_path}")
        new_cell_path = new_cell_path.parent / (new_cell_path.stem + ".npy")
        new_nuclei_path = new_nuclei_path.parent / (new_nuclei_path.stem + ".npy")
        new_cell_paths.append(new_cell_path)
        new_nuclei_paths.append(new_nuclei_path)

    return new_cell_paths, new_nuclei_paths, num_removed, num_total
