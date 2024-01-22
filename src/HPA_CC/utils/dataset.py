from pathlib import Path

class Dataset:
    """Class to index wells and images in a dataset in a reproducible manner.
    Assumes that other than folders listed in folder_excl, all other folders in data_dir are wells,
    and all other folders in each well are images.
    """
    def __init__(self, data_dir, folder_excl=["__pycache__"]):
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.folder_excl = folder_excl
        self.well_list = self.get_wells()
        self.image_list = self.get_images()
    
    def get_wells(self):
        return sorted([d for d in self.data_dir.iterdir() if d.is_dir() and not d.name in self.folder_excl])
    
    def get_images(self):
        image_list = []
        for well in self.well_list:
            image_list += sorted([d for d in well.iterdir() if d.is_dir() and not d.name in self.folder_excl])
        return image_list