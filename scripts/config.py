from pathlib import Path
HOME = Path('/home/ishang/HPA_CC/')
FUCCI_DS_PATH = Path('/data/ishang/FUCCI-dataset-well/')
CCNB1_DS_PATH = Path('/data/ishang/CCNB1-dataset/')
# FUCCI_NAME = "unnormalize_1250_sharp_512_crop_og_res"
# FUCCI_NAME = "minimal_512"
FUCCI_NAME = "minimal"
HPA_DS_PATH = Path('/data/ishang/all_HPA-CC-dataset')

OUTPUT_DIR = Path.home() / "HPA_CC" / "scripts" / "output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()
