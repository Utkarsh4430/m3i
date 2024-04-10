from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
import numpy as np
from tqdm import tqdm
import pandas as pd
import squarify
import os
from PIL import Image
import joblib

import shutil
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea

from matplotlib.cbook import get_sample_data

# Load the embeddings and indices
embeddings = []
indices = []

# Iterate over the files in the 'taiwan_emb' directory
for i in tqdm(os.listdir('taiwan_emb')):
    a = np.load(os.path.join('taiwan_emb', i))
    indices.append(int(i.split('.')[0]))
    embeddings.append(a[0])

# Read the filepaths from 'taiwan_filenamesList.txt'
with open('taiwan_filenamesList.txt') as f:
    filepaths = f.readlines()

filepaths = [i.strip() for i in filepaths]

K=40

# Load the pre-trained KMeans model
kmeans = joblib.load("model.taiwan")
print('len(embeddings): ',len(embeddings))

# Predict the cluster labels for the embeddings
a = kmeans.predict(embeddings)
print(a[:5])

# Find the closest embeddings to the cluster centers
closest, _ = vq(kmeans.cluster_centers_, embeddings)
print(closest)

# Count the number of samples in each cluster
_, counts = np.unique(a, return_counts=True)
df = []
for i in range(K):
    df.append([filepaths[indices[closest[i]]], counts[i]])

df = pd.DataFrame(df, columns=['cluster', 'count'])
print(df.head())

# Set up the treemap plot
x = 0.
y = 0.
width = 4096.
height = 4096.

# Sort the clusters by count in descending order
sorted_c = df.sort_values(by="count", ascending=False)
sorted_c.index = range(0, sorted_c.shape[0])

# Normalize the counts to obtain the sizes of the rectangles
values = squarify.normalize_sizes(sorted_c["count"], width, height)

# Generate the padded rectangles for the treemap
padded_rects = squarify.padded_squarify(values, x, y, width, height)

# Create the treemap plot
plt.ioff()
fig = plt.figure(figsize=(128,128))
ax = fig.add_subplot(1,1,1)
ax.set_xlim((0,width))
ax.set_ylim((0,height))

# Iterate over the clusters and add rectangles and images to the plot
for idx,row in sorted_c.iterrows():
    rect = padded_rects[idx]

    ax.add_patch(Rectangle((rect['x'], rect['y']), rect['dx'], rect['dy'], facecolor="blue"))

    min_v = min(int(rect['dx']), int(rect['dy']))
    if min_v < 10:
        continue

    path = row["cluster"]
    img_raw = Image.open(path)

    # Resize the image to fit the rectangle
    img_resized = img_raw.resize((int(rect['dx'] * 1.5), int(rect['dy'] * 1.5)))

    img = OffsetImage(img_resized)

    # Add the image to the plot
    ab = AnnotationBbox(img, (rect['x'] + rect['dx']/2, rect['y'] + rect['dy']/2), frameon=True)
    ax.add_artist(ab)

    # Annotate with a text box showing the cluster index
    offsetbox = TextArea("Cluster: %d" % (idx))
    ab = AnnotationBbox(offsetbox, (rect['x'] + rect['dx']/2, rect['y'] + rect['dy']/2),
                        bboxprops=dict(boxstyle="sawtooth"))
    ax.add_artist(ab)

# Save the treemap plot as a PDF file
plt.savefig("treemap_taiwan_new.pdf", format="pdf")
