import pandas as pd

from datetime import datetime
import os
import json
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from tqdm import tqdm
import joblib

# Load Taiwan tweet data, including timestamps and image IDs
f = open('/fs/nexus-projects/m3i/data/taiwan_time_id_img.json')
data = json.load(f)

# Load pre-trained KMeans model for clustering
kmeans = joblib.load("model.taiwan")

# Process Taiwan embeddings
embeddings = []
indices = []
for i in tqdm(os.listdir('taiwan_emb')):
    a = np.load(os.path.join('taiwan_emb', i))
    indices.append(int(i.split('.')[0]))
    embeddings.append(a[0])
with open('taiwan_filenamesList.txt') as f:
    filepaths = f.readlines()
filepaths = [i.strip() for i in filepaths]
a = kmeans.predict(embeddings)
store = {filepaths[indices[i]]: a[i] for i in range(len(a))}

# Aggregate Taiwan data by date and cluster assignment
df = {}
for i in data:
    temp = [0]*40
    date_store = {}
    for j in range(0, len(data[i]), 2):
        name = data[i][j]
        date = data[i][j+1][:10]
        key = "/fs/nexus-projects/m3i/data/taiwan_images/" + name
        if key in store:
            date_store.setdefault(date, [0]*40)[store[key]] += 1
    for date in date_store:
        sum_j = sum(date_store[date])
        date_store[date] = [count/sum_j for count in date_store[date]]
        df[f'{date}---{i}'] = date_store[date]

# Repeat the process for Hong Kong data
f = open('/fs/nexus-projects/m3i/data/hk_time_id_img.json')
data = json.load(f)
embeddings = []
indices = []
for i in tqdm(os.listdir('hongkong_emb')):
    a = np.load(os.path.join('hongkong_emb', i))
    indices.append(int(i.split('.')[0]))
    embeddings.append(a[0])
with open('hongkong_filenamesList.txt') as f:
    filepaths = f.readlines()
filepaths = [i.strip() for i in filepaths]
a = kmeans.predict(embeddings)
store = {filepaths[indices[i]]: a[i] for i in range(len(a))}
df2 = {}
for i in data:
    temp = [0]*40
    date_store = {}
    for j in range(0, len(data[i]), 2):
        name = data[i][j]
        date = data[i][j+1][:10]
        key = "/fs/nexus-projects/m3i/data/hongkong_images/" + name
        if key in store:
            date_store.setdefault(date, [0]*40)[store[key]] += 1
    for date in date_store:
        sum_j = sum(date_store[date])
        date_store[date] = [count/sum_j for count in date_store[date]]
        df2[f'{j}---{i}'] = date_store[date]

# Compare the cluster distributions between Taiwan and Hong Kong data on matching dates
sum_array = []
plot_values = {}
for i in df2:
    for j in df:
        if i[:10] == j[:10]:  # Matching dates
            temp = datetime(int(i[:4]), int(i[5:7]), int(i[8:10]))
            similarity = 1 - cosine(df2[i], df[j])
            plot_values.setdefault(temp, []).append(similarity)
            sum_array.append(similarity)

# Prepare and save the comparison results
date_csv = pd.DataFrame([(date, len(values), sum(values)/len(values)) for date, values in plot_values.items()], columns=['Date', 'Number of Datapoints', 'Average value'])
date_csv.to_csv('taiwan-train-hk-fit.csv', index=False)

# Generate a plot of average similarity over time
x_plot, y_plot = zip(*sorted(plot_values.items()))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=90))
plt.plot(x_plot, [sum(values)/len(values) for values in y_plot])
plt.gcf().autofmt_xdate()
plt.savefig('Taiwan-train-hk-fit-day-graph.png')
