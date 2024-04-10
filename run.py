# Set up environment variable for the Transformers library to specify a custom cache location.
import os
os.environ['TRANSFORMERS_CACHE'] = '/fs/clip-projects/m3i/campaigns/new-campaigns'

# Import necessary libraries: PIL for image processing, numpy for numerical operations,
# transformers for loading the CLIP model, torchvision for image transformations,
# torch for tensor operations, and tqdm for a progress bar.
from PIL import Image
import numpy as np
from transformers import AutoProcessor, CLIPModel
from torchvision import transforms
import torch
from tqdm import tqdm

# Determine the device (CUDA GPU or CPU) for running the model.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the CLIP model and move it to the determined device.
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model = model.to(device)

# Initialize the processor for the CLIP model for handling image inputs.
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Read the list of image file paths from a text file.
with open('filenamesList.txt') as f:
    filepaths = f.readlines()

# Set the batch size for processing images. Here, set to 1 for sequential processing.
batch_size = 1

# Define the transformation to convert images to tensors.
transform = transforms.Compose([transforms.ToTensor()])

# List the already processed embedding files to avoid reprocessing.
temp_files = os.listdir('./embeddings')

# Process images in batches to generate embeddings.
for i in tqdm(range(0, len(filepaths), batch_size)):
    # Skip already processed files.
    if f'{i}.npy' in temp_files:
        continue
    try:
        # Calculate the range for the current batch.
        high = min(i+batch_size, len(filepaths))
        filepaths_new = filepaths[i:high]

        # Load and transform the images.
        images_arr = [Image.open(i.strip()) for i in filepaths_new]
        images_arr = [transform(image).to(device) for image in images_arr]

        # Process the images through the CLIP processor.
        image_processor = processor(images=images_arr, return_tensors="pt")
        del images_arr  # Free up memory by deleting the original image array.

        # Move pixel values to the correct device.
        image_processor['pixel_values'] = image_processor['pixel_values'].to(device)

        # Calculate the image features/embeddings using the CLIP model.
        image_features = model.get_image_features(**image_processor)
        del image_processor  # Free up memory.

        # Convert the tensor of image features to a numpy array and save to disk.
        image_features = torch.Tensor.numpy(image_features, force=True)
        np.save(f'./embeddings/{i}', image_features)
    except:
        # Log any failures during the process.
        print(f'embeddings/{i} failed')
