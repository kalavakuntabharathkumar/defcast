from pathlib import Path
import pandas as pd
VALID_LABELS={"normal","defect"}
def validate_image_manifest(manifest: pd.DataFrame, image_root: str | Path):
    required={"filename","label"}
    if not required.issubset(manifest.columns): raise ValueError(f"Missing columns: {sorted(required-set(manifest.columns))}")
    if manifest.empty: raise ValueError("Manifest must not be empty")
    if manifest["filename"].isna().any() or manifest["label"].isna().any(): raise ValueError("Null filename or label")
    invalid=set(manifest["label"].astype(str))-VALID_LABELS
    if invalid: raise ValueError(f"Unknown labels: {sorted(invalid)}")
    root=Path(image_root)
    missing=[name for name in manifest["filename"].astype(str) if not (root/name).is_file()]
    if missing: raise FileNotFoundError(f"Missing image files: {missing[:5]}")
    if manifest["filename"].duplicated().any(): raise ValueError("Duplicate image filenames")
    return {"rows":len(manifest),"class_counts":manifest["label"].value_counts().to_dict(),"missing":0}

def validate_demand_frame(frame: pd.DataFrame):
    required={"date","demand"}
    if not required.issubset(frame.columns): raise ValueError(f"Missing columns: {sorted(required-set(frame.columns))}")
    if frame.empty or frame["demand"].isna().any(): raise ValueError("Demand data is empty or contains nulls")
    if (frame["demand"]<0).any(): raise ValueError("Demand must be non-negative")
    return True
