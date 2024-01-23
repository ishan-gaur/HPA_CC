from pathlib import Path
from config import HPA_DS_PATH
from HPA_CC.utils.dataset import Dataset

is_well = lambda f: len(f.name.split("_")) == 3
well_folders = list(filter(is_well, HPA_DS_PATH.iterdir()))
wells = set(['_'.join(f.name.split('_')[:-1]) for f in well_folders])
for well in wells:
    image_folders = list(filter(lambda f: f.name.startswith(well), well_folders))
    well_folder = HPA_DS_PATH / well
    try:
        well_folder.mkdir()
    except FileExistsError:
        print(f"{well_folder} already exists")
    for folder in image_folders:
        folder.rename(well_folder / folder.name)

hpa_ds = Dataset(HPA_DS_PATH)
print(f"Number of wells: {len(hpa_ds.well_list)}")
print(f"Number of images: {len(hpa_ds.image_list)}")