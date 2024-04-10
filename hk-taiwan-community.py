import pandas as pd
from datetime import datetime
import os
import json
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from tqdm import tqdm
import shutil
from sklearn.metrics import silhouette_score
import joblib

# Load tweet data and community assignments for Taiwanese tweets
f = open('/fs/nexus-projects/m3i/data/taiwan_time_id_img.json')
data = json.load(f)
community = open('/fs/nexus-projects/m3i/data/louvain/taiwan_retweet_all.json')
community = json.load(community)

# Identify the top 40 communities by size
comms = sorted([(len(community[i]), i) for i in community], key=lambda tup: tup[0], reverse=True)[:40]

# Load embeddings for Taiwanese campaign images
embeddings = []
indices = []
for i in tqdm(os.listdir('taiwan_emb')):
    a = np.load(os.path.join('taiwan_emb', i))
    indices.append(int(i.split('.')[0]))
    embeddings.append(a[0])

# Prepare file paths for further processing
with open('taiwan_filenamesList.txt') as f:
    filepaths = f.readlines()
filepaths = [i.strip() for i in filepaths]

# Load a pre-trained KMeans model
kmeans = joblib.load("model.taiwan")
a = kmeans.predict(embeddings)

# Map each filename to its predicted cluster
store = {filepaths[indices[i]]: a[i] for i in range(len(a))}

# Prepare data from a reference campaign for comparison
campaign_name = 'china_052020'
campaign_path = f'/fs/nexus-projects/m3i/campaigns/{campaign_name}/'
references = [f'{campaign_path}china_052020_tweets_csv_unhashed.csv']
collect = [pd.read_csv(i) for i in references]
reference = pd.concat(collect)
del collect  # Free memory
reference_store = {str(i[0]): i[13][:10] for i in reference.values}

# Load and predict clusters for another campaign's embeddings
embeddings_campaign = []
indices_campaign = []
for i in tqdm(os.listdir(f'{campaign_path}embeddings')):
    a = np.load(os.path.join(f'{campaign_path}embeddings', i))
    indices_campaign.append(int(i.split('.')[0]))
    embeddings_campaign.append(a[0])
a_campaign = kmeans.predict(embeddings_campaign)

# Aggregate cluster data for the comparative campaign
store_campaign = {}
for i in range(len(a_campaign)):
    filename = filepaths_campaign[indices_campaign[i]]
    user = filename.split('/')[-2]
    tweetid = filename.split('/')[-1].split('-')[0]
    if tweetid in reference_store:
        uniqueid = f"{reference_store[tweetid]}---{user}"
        if uniqueid not in store_campaign:
            store_campaign[uniqueid] = [0]*K
        store_campaign[uniqueid][a_campaign[i]] += 1
for key in store_campaign:
    sum_values = sum(store_campaign[key])
    store_campaign[key] = [value/sum_values for value in store_campaign[key]]

# Prepare directories for output
os.makedirs('taiwan-china2020-retweet-all', exist_ok=True)

# Analyze and visualize data for each of the top communities
for comm in comms:
    allowed = set(community[comm[1]])
    df = {}
    for i in data:
        if i in allowed:
            # Initialize cluster count data
            date_store = {}
            for j in range(0, len(data[i]), 2):
                name, date = data[i][j], data[i][j+1][:10]
                key = f"/fs/nexus-projects/m3i/data/taiwan_images/{name}"
                if key in store:
                    date_store.setdefault(date, [0]*K)[store[key]] += 1
            # Normalize cluster counts by date
            for date in date_store:
                sum_j = sum(date_store[date])
                date_store[date] = [value/sum_j for value in date_store[date]]
                df[f'{date}---{i}'] = date_store[date]

    # Calculate similarities between Taiwanese and reference campaign data
    sum_array = []
    plot_values = {}
    for taiwan_date in tqdm(df):
        for campaign_date in store_campaign:
            if taiwan_date[:10] == campaign_date[:10]:
                temp_date = datetime(int(taiwan_date[:4]), int(taiwan_date[5:7]), int(taiwan_date[8:10]))
                plot_values.setdefault(temp_date, []).append(1-cosine(df[taiwan_date], store_campaign[campaign_date]))
                sum_array.append(1-cosine(df[taiwan_date], store_campaign[campaign_date]))

    # Aggregate and save the data, then generate plots for each community
    os.makedirs(f'taiwan-china2020-retweet-all/{comm[1]}', exist_ok=True)
    date_csv = pd.DataFrame([(date, len(values), sum(values)/len(values)) for date, values in plot_values.items()], columns=['Date', 'Number of Datapoints', 'Average value'])
    date_csv.to_csv(f'taiwan-china2020-retweet-all/{comm[1]}/datewise.csv', index=False)
    # Plotting logic omitted for brevity
