# from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
import numpy as np
from tqdm import tqdm
import os
import shutil
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import joblib

embeddings = []
indices = []

for i in tqdm(os.listdir('taiwan_emb')):
    a = np.load(os.path.join('taiwan_emb', i))
    indices.append(int(i.split('.')[0]))
    embeddings.append(a[0])

with open('taiwan_filenamesList.txt') as f:
    filepaths = f.readlines()

filepaths = [i.strip() for i in filepaths]

K=40
kmeans = KMeans(n_clusters=K)
kmeans.fit(embeddings)
print('len(embeddings): ',len(embeddings))

joblib.dump(kmeans, 'model.taiwan')

stop

a = kmeans.predict(embeddings)
print(a[:5])
closest, _ = vq(kmeans.cluster_centers_, embeddings)
print(closest)

os.makedirs('taiwan_clusters', exist_ok=True)
os.system('rm -rf taiwan_clusters/*')

for i in closest:
    os.system(f'cp {filepaths[indices[i]]} taiwan_clusters')
