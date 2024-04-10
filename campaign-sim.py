import os
import json
from scipy.spatial.distance import cosine
import numpy as np
from tqdm import tqdm
import os

# Load JSON data mapping tweet IDs to image paths
f = open('/fs/nexus-projects/m3i/data/hk_tweetid_to_imagepath.json')
data = json.load(f)

# Initialize lists to store embeddings and their indices
embeddings = []
indices = []

# Read file paths from a text file into a list
with open('hongkong_filenamesList.txt') as f:
    filepaths = f.readlines()
filepaths = [i.strip() for i in filepaths]

# Collect the list of precomputed embeddings files
temp_files = set(os.listdir('hongkong_emb'))
hk_user_emb = []

# Process each user in the data
for i in tqdm(data):
    temp = []
    # For each image associated with the user, find its embedding if available
    for j in data[i]:
        key = "/fs/nexus-projects/m3i/data/hongkong_images/" + j
        try:
            index_value = filepaths.index(key)
        except ValueError:
            index_value = -1

        if index_value != -1 and f'{index_value}.npy' in temp_files:
            a = np.load(os.path.join('hongkong_emb', f'{index_value}.npy'))[0]
            temp.append(a)
    if len(temp):
        temp = np.array(temp).mean(axis=0)
        assert temp.shape == (768,)
        hk_user_emb.append(temp)

# Load file paths for a second campaign
second_campaign = '/fs/nexus-projects/m3i/campaigns/china_052020/filenamesList.txt'
embeddings_path = '/fs/nexus-projects/m3i/campaigns/china_052020/embeddings'

with open(second_campaign) as f:
    filepaths = f.readlines()
filepaths = [i.strip() for i in filepaths]

# Initialize a dictionary to store user embeddings
store = {}
embedding_list = set(os.listdir(embeddings_path))

# Process each file path, load embeddings, and aggregate by user
for i in tqdm(range(len(filepaths))):
    if f'{i}.npy' not in embedding_list:
        continue
    user = filepaths[i].split('/')[-2]
    a = np.load(os.path.join(embeddings_path, f'{i}.npy'))[0]
    if user in store:
        store[user].append(a)
    else:
        store[user] = [a]

# Average embeddings per user and assert the embedding size
final_user_embeddings = []
for i in tqdm(store):
    arr = np.array(store[i]).mean(axis=0)
    assert arr.shape == (768,)
    final_user_embeddings.append(arr)

# Compute cosine similarity between each pair of embeddings from the two campaigns
sum_array = []
for i in tqdm(hk_user_emb):
    for j in final_user_embeddings:
        sum_array.append(cosine(i, j))

print('Average Similarity: ', sum(sum_array)/len(sum_array))
