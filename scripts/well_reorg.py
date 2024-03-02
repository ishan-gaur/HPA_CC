import os
import re
import shutil
import argparse
from glob import glob
from pathlib import Path


parser = argparse.ArgumentParser(description='Reorganize FUCCI dataset into wells')
parser.add_argument('--data', type=str, default='/data/ishang/FUCCI-dataset/')
parser.add_argument('--output', type=str, default='/data/ishang/FUCCI-data/')
parser.add_argument('--copy', action='store_true', default=False)
parser.add_argument('--field', action='store_true', default=False)
parser.add_argument('--overview', action='store_true', default=False)
parser.add_argument('--tilescan', action='store_true', default=False)
parser.add_argument('--all', action='store_true', default=False)
parser.add_argument('--check', action='store_true', default=False)
parser.add_argument('--num-channels', type=int, default=4)

image_files = ['Geminin.png', 'microtubule.png', 'nuclei.png', 'CDT1.png']
image_files = ['microtubule.png', 'nuclei.png', 'cyclinb1.png']

args = parser.parse_args()

if not Path(args.data).is_absolute():
    raise ValueError("Data directory should be an absolute path")

if not Path(args.output).is_absolute():
    raise ValueError("Output directory should be an absolute path")
if not Path(args.output).exists():
    os.mkdir(args.output)

args.data = Path(args.data)
args.output = Path(args.output)

if args.copy:
    import shutil
    print(f'Copying all folders\' images from {str(args.data)} to {str(args.output)}')

    for folder in args.data.iterdir():
        if not folder.is_dir():
            continue

        skip = False
        for file_name in image_files:
            if not (folder / file_name).exists():
                print(f'Folder {folder.name} is missing {file_name}')
                skip = True
                break
        
        if skip: 
            continue
        
        if not (args.output / folder.name).exists():
            os.mkdir(args.output / folder.name)
            for file_name in image_files:
                shutil.copy(src=(folder / file_name), dst=(args.output / folder.name))
        else:
            print(f'Folder {folder.name} already exists in {args.output}')
            for file_name in image_files:
                if not (args.output / folder.name / file_name).exists():
                    shutil.copy(src=(folder / file_name), dst=(args.output / folder.name))
                    print(f'\tCopied {file_name} to {args.output / folder.name}')
            continue

def group_by_well(field_folders, extract_well_fn, well_name_fn):
    well_names = set()
    for folder in field_folders:
        well_names.add(extract_well_fn(folder))
    well_names = list(well_names)

    field_names = []
    for well in well_names:
        well_folder = args.output / well_name_fn(well)
        if not well_folder.exists():
            os.mkdir(well_folder)
        for folder in field_folders:
            if extract_well_fn(folder) == well:
                folder = Path(folder)
                field_names.append(folder.name)
                shutil.copytree(src=folder, dst=well_folder / Path(folder).name)

    assert len(field_names) == len(set(field_names)), "Duplicate field names when grouping by well"
    assert len(field_names) == len(field_folders), f"Not all field folders were moved, new fields {len(field_names)} != src fields {len(field_folders)}"

if args.field or args.all:
    field_folders = glob(f"{args.data}/field--X[0-9][0-9]--Y[0-9][0-9]_image--*--U[0-9][0-9]--V[0-9][0-9]--*/")
    extract_well = lambda x: re.search(r'--U([0-9][0-9])--V([0-9][0-9])--', x).groups()
    well_name = lambda x: f'chamber--U{x[0]}--V{x[1]}'
    group_by_well(field_folders, extract_well, well_name)

if args.overview or args.all:
    overview_folders = glob(f"{args.data}/Overview [0-9]_Image [0-9][0-9]--Stage[0-9][0-9]/")
    extract_well = lambda x: re.search(r'Overview ([0-9])_Image ([0-9][0-9])--Stage[0-9][0-9]', x).groups()
    well_name = lambda x: f'overview{x[0]}--image{x[1]}'
    group_by_well(overview_folders, extract_well, well_name)

if args.tilescan or args.all:
    tilescan_export_folders = glob(f"{args.data}/TileScan [0-9] - export_TileScan 1_[A-Z]*/")
    extract_well = lambda x: re.search(r'TileScan ([0-9]) - export_TileScan [0-9]_([A-Z][0-9]+)', x).groups()
    well_name = lambda x: f'tilescan{x[0]}--{x[1]}'
    group_by_well(tilescan_export_folders, extract_well, well_name)

    # These bottom two work because the wells are only of the form [A-Z][0-9], I'll probably have to use a regex going forward for wells like [A-z][0-9][0-9]
    tilescan_short_folders = glob(f"{args.data}/TileScan [0-9]_[A-Z][0-9] Region*/")
    extract_well = lambda x: re.search(r'TileScan ([0-9])_([A-Z][0-9]+)', x).groups()
    well_name = lambda x: f'tilescan{x[0]}--{x[1]}'
    group_by_well(tilescan_short_folders, extract_well, well_name)

    tilescan_long_folders = glob(f"{args.data}/TileScan [0-9]_TileScan [0-9]_[A-Z]*Region*/")
    extract_well = lambda x: re.search(r'TileScan ([0-9])_TileScan [0-9]_([A-Z][0-9]+)', x).groups()
    well_name = lambda x: f'tilescan{x[0]}--{x[1]}'
    group_by_well(tilescan_long_folders, extract_well, well_name)

if args.check or args.all:
    orig_img_count, orig_folder_count = 0, 0
    for folder in args.data.iterdir():
        if not folder.is_dir():
            continue
        orig_folder_count += 1
        for image in folder.iterdir():
            if not image.is_dir():
                orig_img_count += 1

    new_img_count, new_folder_count = 0, 0
    well_count = 0
    for well in args.output.iterdir():
        if well.is_dir():
            well_count += 1
            for folder in well.iterdir():
                if folder.is_dir():
                    new_folder_count += 1
                    files_per_folder = 0
                    for image in folder.iterdir():
                        if not image.is_dir():
                            new_img_count += 1

    assert orig_img_count % args.num_channels == 0, f"Number of images in the original folder is not a multiple of the channel count {args.num_channels}"
    assert new_img_count % args.num_channels == 0, f"Number of images in the new folder is not a multiple of the channel count {args.num_channels}"
    print(f'Original count: {orig_img_count}, New count: {new_img_count}')
    print(f'Original folders: {orig_folder_count}, New folders: {new_folder_count}')
    print(f'Number of wells found: {well_count}')
