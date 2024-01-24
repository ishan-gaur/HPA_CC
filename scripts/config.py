from pathlib import Path
FUCCI_DS_PATH = Path('/data/ishang/FUCCI-dataset-well')

OUTPUT_DIR = Path.cwd() / "scripts" / "output"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()