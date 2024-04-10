import os
from pathlib import Path

# Define the input path where the original campaign media files are stored
input_path = '/fs/clip-projects/m3i/campaigns/new-campaigns/2019_08/china_082019_2/china_082019_2_tweet_media_unhashed'

# First loop through each directory in the input path
for i in os.listdir(input_path):
    filepath1 = os.path.join(input_path, i)
    # For each directory, loop through each file
    for j in os.listdir(filepath1):
        # Split the filename to remove file extension
        name = i.split('.')[0]
        filepath2 = os.path.join(filepath1, j)
        # Unzip each file in its parent directory
        os.system(f'unzip {filepath2} -d {filepath1}')

# Loop again through each item in the input directory
for i in os.listdir(input_path):
    filepath1 = os.path.join(input_path, i)
    # Check if the item is a zip file
    if '.zip' in filepath1:
        name = i.split('.')[0]
        # Unzip the file to a directory named after the file (excluding extension) while excluding mp4 files, then delete the zip file
        os.system(f'unzip {filepath1} -d {name} -x *.mp4')
        os.system(f'rm -rf {filepath1}')

p = Path("/fs/clip-projects/m3i/campaigns/new-campaigns/2019_08/china_082019_1/china_082019_1_tweet_media_unhashed")

filenamesList = []

# Collect all .jpg files in the directory and subdirectories
for i in p.glob('**/*.jpg'):
    filenamesList.append(i)

# Collect all .png files in the directory and subdirectories
for i in p.glob('**/*.png'):
    filenamesList.append(i)

# Write the paths of all collected image files to a text file
with open('filenamesList.txt', 'w') as f:
    for i in filenamesList:
        f.write(f'{i}\n')

print(len(filenamesList))
