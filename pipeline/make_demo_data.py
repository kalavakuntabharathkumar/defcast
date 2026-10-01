"""Generate tiny synthetic geometry images for pipeline smoke tests only."""
from pathlib import Path
import csv, random
from PIL import Image, ImageDraw

def make_data(root="data/generated", count=80, seed=7):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); rng=random.Random(seed); rows=[]
    for i in range(count):
        label="defect" if i%2 else "normal"
        img=Image.new("RGB",(64,64),"white"); draw=ImageDraw.Draw(img)
        draw.rectangle((10,10,53,53),outline="black",width=2)
        if label=="defect":
            x,y=rng.randint(15,45),rng.randint(15,45); draw.line((x,y,x+8,y+8),fill="red",width=3)
        else: draw.ellipse((25,25,38,38),outline="blue",width=2)
        filename=f"image_{i:04d}.png"; img.save(root/filename); rows.append({"filename":filename,"label":label})
    with (root/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["filename","label"]); writer.writeheader(); writer.writerows(rows)
    return root/"manifest.csv"
if __name__=="__main__": print(f"Created demo manifest: {make_data()}")
