from pathlib import Path
FUCCI_DS_PATH = Path('/data/ishang/FUCCI-dataset-well')
FUCCI_NAME = "unnormalize_1250_sharp_512_crop_og_res"
HPA_DS_PATH = Path('/data/ishang/all_HPA-CC-dataset')

OUTPUT_DIR = Path.cwd() / "output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()
