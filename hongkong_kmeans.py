from sklearn.cluster import MiniBatchKMeans as KMeans
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

for i in tqdm(os.listdir('hongkong_emb')):
    a = np.load(os.path.join('hongkong_emb', i))
    indices.append(int(i.split('.')[0]))
    embeddings.append(a[0])

with open('hongkong_filenamesList.txt') as f:
    filepaths = f.readlines()

filepaths = [i.strip() for i in filepaths]

Sum_of_squared_distances = []
silhouette_avg = []

# K = range(5,101,5)
# for num_clusters in tqdm(K) :
#  kmeans = KMeans(n_clusters=num_clusters, batch_size=2048)
#  kmeans.fit(embeddings)
#  Sum_of_squared_distances.append(kmeans.inertia_)
#  silhouette_avg.append(silhouette_score(embeddings, kmeans.labels_))

# plt.plot(K,Sum_of_squared_distances,'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Sum of squared distances/Inertia')
# plt.title('Elbow Method For Optimal k')
# plt.savefig('elbow-hongkong.jpg')
# plt.clf()
# plt.plot(list(K),silhouette_avg,'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Silhouette score')
# plt.title('Silhouette analysis For Optimal k')
# plt.savefig('silhouette-hongkong.jpg')

K=40
kmeans = KMeans(n_clusters=K, batch_size=2048)
kmeans.fit(embeddings)
print('len(embeddings): ',len(embeddings))
joblib.dump(kmeans, "model.hk")

stop


a = kmeans.predict(embeddings)
print(a[:5])

print(len(a))

asdasd
closest, _ = vq(kmeans.cluster_centers_, embeddings)
print(closest)

os.makedirs('hongkong_clusters', exist_ok=True)
os.system('rm -rf hongkong_clusters/*')

for i in closest:
    os.system(f'cp {filepaths[indices[i]]} hongkong_clusters')
