"""Minimal CNN training entry point; install PyTorch separately to run."""
from pathlib import Path
import pandas as pd
from .data_validation import validate_image_manifest
from .experiment import log_metrics

def train(data_root="data/generated", epochs=5):
    if epochs < 1: raise ValueError("epochs must be >= 1")
    try:
        import torch
        from torch import nn
        from torch.utils.data import Dataset, DataLoader
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("PyTorch is optional. Install a compatible torch build, then retry.") from exc
    root=Path(data_root); manifest=pd.read_csv(root/"manifest.csv")
    validate_image_manifest(manifest,root)
    class ImageDataset(Dataset):
        def __len__(self): return len(manifest)
        def __getitem__(self,index):
            row=manifest.iloc[index]
            with Image.open(root/row.filename) as source: image=source.convert("RGB").resize((64,64))
            x=np.asarray(image,dtype=np.float32).transpose(2,0,1)/255.0
            return torch.tensor(x),torch.tensor(1 if row.label=="defect" else 0,dtype=torch.long)
    loader=DataLoader(ImageDataset(),batch_size=16,shuffle=True)
    model=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(32,2))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
    total_loss=0.0
    for _ in range(epochs):
        model.train()
        for x,y in loader:
            opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step(); total_loss+=float(loss.item())
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in loader:
            logits=model(x); correct+=int((logits.argmax(1)==y).sum()); total+=len(y)
    Path("artifacts/models").mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),"artifacts/models/defect_cnn.pt")
    metrics={"epochs":epochs,"train_accuracy":correct/max(total,1),"mean_batch_loss":total_loss/max(len(loader)*epochs,1)}
    log_metrics("defect_cnn_demo",metrics); return metrics
if __name__=="__main__": print(train())
