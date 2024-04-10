import pandas as pd
import os
import json
from sklearn.metrics import pairwise_distances

# Load JSON data containing mappings from tweet IDs to image paths
f = open('/fs/nexus-projects/m3i/data/hk_tweetid_to_imagepath.json')
data = json.load(f)

from sklearn.cluster import MiniBatchKMeans as KMeans
import numpy as np
from tqdm import tqdm

# Load embeddings and their indices from numpy files
embeddings = []
indices = []
for i in tqdm(os.listdir('hongkong_emb')):
    a = np.load(os.path.join('hongkong_emb', i))
    indices.append(int(i.split('.')[0]))
    embeddings.append(a[0])

# Read file paths from a text file and clean them
with open('hongkong_filenamesList.txt') as f:
    filepaths = f.readlines()
filepaths = [i.strip() for i in filepaths]

# Perform KMeans clustering on the embeddings
K = 40
kmeans = KMeans(n_clusters=K, batch_size=2048)
kmeans.fit(embeddings)
a = kmeans.predict(embeddings)

# Map filenames to their predicted cluster
store = {}
for i in range(len(a)):
    prediction = a[i]
    filename = filepaths[indices[i]]
    store[filename] = prediction

# Prepare data for analysis, counting occurrences of each cluster per user
df = []
df2 = []
counter = 0  # Counter for images not found in the store
for i in data:
    temp = [0]*K  # Initialize a list to count occurrences in each cluster
    for j in data[i]:
        key = "/fs/nexus-projects/m3i/data/hongkong_images/" + j
        if key in store:
            temp[store[key]] += 1
        else:
            counter += 1

    # Normalize the cluster counts by the total counts for a user
    temp_sum = sum(temp)
    if temp_sum:
        temp = [j/temp_sum for j in temp]
        df.append(temp)
        temp = [i] + temp
        df2.append(temp)

# Calculate pairwise cosine similarity between users' cluster distributions
dist_out = 1 - pairwise_distances(df, metric="cosine")
dists = []
for i in range(len(dist_out)-1):
    for j in range(i+1, len(dist_out)):
        dists.append(abs(dist_out[i][j]))

# Calculate pairwise Euclidean distance between users' cluster distributions
dist_out = pairwise_distances(df, metric="euclidean")
dists = []
for i in range(len(dist_out)-1):
    for j in range(i+1, len(dist_out)):
        dists.append(abs(dist_out[i][j]))

# Convert the data into a pandas DataFrame and save it to a CSV file
df2 = pd.DataFrame(df2)
df2.to_csv('hk-user-cluster.csv', index=False)

# The code calculates the similarity between users based on the distribution of their images across the clusters. It also counts the number of images that could not be found in the clustering process.
