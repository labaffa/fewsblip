import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import zipfile
from pathlib import Path
import csv


p = "/media/gio/Samsung_T5/dummy_data/phase2/telecatch_data/haiti_20240305.zip"

with zipfile.ZipFile(p, mode="r") as archive:
    for f in archive.namelist():
        archive.read(f)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def load_demo_image(image_size,device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   

    w,h = raw_image.size
    # display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image




image_size = 384
image = load_demo_image(image_size=image_size, device=device)

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    
model = blip_decoder(pretrained='./checkpoints/model_base_caption_capfilt_large.pth', image_size=image_size, vit='base')
model.eval()
model = model.to(device)
out_path = "./haiti_test.tsv"

with torch.no_grad():
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    # beam search
    # caption = model.generate(image, sample=False, num_beams=1, max_length=20, min_length=5) 
    # nucleus sampling
    with open(out_path, "w") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(("date", "channel_id", "message_id", "caption_nucleus"))
        with zipfile.ZipFile(p, mode="r") as archive:
            img_number = len(list(archive.namelist()))
            for i, f in enumerate(archive.namelist()):
                if f.startswith('media/'):
                    try:
                        channel_id, message_id = Path(f).stem.split("_")
                        raw_image = Image.open(archive.open(f)).convert('RGB') 
                        image = transform(raw_image).unsqueeze(0).to(device) 
                        nucleus = model.generate(image, sample=True, top_p=0.9, max_length=40, min_length=5)
                        # beam = model.generate(image, sample=False, num_beams=1, max_length=20, min_length=5) 
                        row = (Path(f).parts[1], channel_id, message_id, nucleus[0])
                        writer.writerow(row)
                        print(f'{i+1}/{img_number} - {Path(f).parts[1]} - {channel_id}_{message_id}: ', nucleus[0])
                    except Exception as e:
                        print(f'{Path(f).parts[1]} - {channel_id}_{message_id}: ERROR {str(e)}')