cache_dir = '/fs/clip-projects/m3i/campaigns/new-campaigns'
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoTokenizer, Blip2Model
from torchvision import transforms
import torch
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)
model = model.to(device)

# Get the image features
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)

with open('hongkong_filenamesList.txt') as f:
    filepaths = f.readlines()

batch_size = 1

for i in tqdm(range(0, len(filepaths), batch_size)):
    try:
        high = min(i+batch_size, len(filepaths))
        filepaths_new = filepaths[i:high]
        images_arr = [Image.open(i.strip()) for i in filepaths_new]
        image_processor = processor(images=images_arr, return_tensors="pt")
        del images_arr
        image_processor['pixel_values'] = image_processor['pixel_values'].to(device)
        # Calculating embeddings
        image_features = model.get_image_features(**image_processor).last_hidden_state.mean(dim=1)
        del image_processor
        print(image_features.shape)
        image_features = torch.Tensor.numpy(image_features, force=True)
        np.save(f'/fs/clip-projects/m3i/data/hongkong_emb_blip/{i}', image_features)
    except Exception as e:
        print(e)
        print(f'{i} failed')