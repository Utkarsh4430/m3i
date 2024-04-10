import pandas as pd
from datetime import datetime
import os
import json
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
f = open('/fs/nexus-projects/m3i/data/taiwan_time_id_img.json')
data = json.load(f)

import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
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

kmeans = joblib.load("model.taiwan")

print('len(embeddings): ',len(embeddings))
a = kmeans.predict(embeddings)

store = {}

for i in range(len(a)):
    prediction = a[i]
    filename = filepaths[indices[i]]
    store[filename] = prediction

df = {}
counter = 0
for i in data:
    temp = [0]*K
    date_store = {}
    for j in range(0,len(data[i]),2):
        name = data[i][j]
        date = data[i][j+1]
        date = date[:10]
        key = "/fs/nexus-projects/m3i/data/taiwan_images/" + data[i][j]
        if key in store:
            if date not in date_store:
                date_store[date] = [0]*K

            date_store[date][store[key]] += 1

    for j in date_store:
        sum_j = sum(date_store[j])
        date_store[j] = [xx/sum_j for xx in date_store[j]]
        df[f'{j}---{i}'] = date_store[j]


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

K=40
print('len(embeddings): ',len(embeddings))
a = kmeans.predict(embeddings)

store = {}

for i in range(len(a)):
    prediction = a[i]
    filename = filepaths[indices[i]]
    store[filename] = prediction

df2 = {}
counter = 0
for i in data:
    temp = [0]*K
    date_store = {}
    for j in range(0,len(data[i]),2):
        name = data[i][j]
        date = data[i][j+1]
        date = date[:10]
        key = "/fs/nexus-projects/m3i/data/hongkong_images/" + data[i][j]
        if key in store:
            if date not in date_store:
                date_store[date] = [0]*K

            date_store[date][store[key]] += 1

    for j in date_store:
        sum_j = sum(date_store[j])
        date_store[j] = [xx/sum_j for xx in date_store[j]]
        df2[f'{j}---{i}'] = date_store[j]

sum_array = []
plot_values = {}
for i in tqdm(df2):
    for j in df:

        if i[:10]==j[:10]:
            # print(i[:10])
            temp = datetime(int(i[:4]), int(i[5:7]), int(i[8:10]))
            if temp not in plot_values:
                plot_values[temp] = []

            plot_values[temp].append(1-cosine(df2[i],df[j]))
            sum_array.append(1-cosine(df2[i],df[j]))

x_plot = []
y_plot = []
for i in plot_values:
    x_plot.append(i)
    y_plot.append(sum(plot_values[i])/len(plot_values[i]))

date_csv = []
for i in range(len(x_plot)):
    date_csv.append([x_plot[i],len(plot_values[x_plot[i]]), y_plot[i]])

date_csv = pd.DataFrame(date_csv, columns=['Date', 'Number of Datapoints', 'Average value'])
date_csv.to_csv(f'taiwan-train-hk-fit.csv', index=False)

x_plot, y_plot = zip(*sorted(zip(x_plot, y_plot)))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=90))

plt.plot(x_plot, y_plot)
plt.gcf().autofmt_xdate()

plt.savefig(f'Taiwan-train-hk-fit-day-graph.png')

print('Average Similarity: ', sum(sum_array)/len(sum_array))
