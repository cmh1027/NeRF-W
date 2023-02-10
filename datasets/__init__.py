from .blender import BlenderDataset
from .llff import LLFFDataset
from .phototourism import PhototourismDataset, PhototourismOptimizeDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'phototourism_optimize': PhototourismOptimizeDataset, 
                'phototourism': PhototourismDataset}