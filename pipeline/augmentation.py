"""Guardrail interface for externally generated synthetic image augmentation.

Stable Diffusion generation is intentionally not automated here: synthetic images
must be reviewed, licensed appropriately, and labelled before entering training.
"""
from pathlib import Path
from PIL import Image

def validate_synthetic_images(paths, expected_size=(64,64)):
    approved=[]
    for item in paths:
        path=Path(item)
        if not path.is_file(): raise FileNotFoundError(path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != expected_size: raise ValueError(f"Unexpected size for {path}: {image.size}")
        approved.append(path)
    return approved
