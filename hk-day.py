import pandas as pd
from datetime import datetime
import os
import json
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import joblib

# Load JSON data mapping Taiwanese tweet IDs to image paths and timestamps
f = open('/fs/clip-projects/m3i/data/taiwan_time_id_img.json')
data = json.load(f)

# Load embeddings for Taiwanese campaign images
embeddings = []
indices = []
for i in tqdm(os.listdir('taiwan_emb')):
    a = np.load(os.path.join('taiwan_emb', i))
    indices.append(int(i.split('.')[0]))
    embeddings.append(a[0])

# Read and clean file paths from a text file
with open('taiwan_filenamesList.txt') as f:
    filepaths = f.readlines()
filepaths = [i.strip().replace('nexus','clip') for i in filepaths]

# Load the pre-trained KMeans model
K=40
kmeans = joblib.load("model.taiwan")
a = kmeans.predict(embeddings)

# Map filenames to their predicted cluster
store = {}
for i in range(len(a)):
    prediction = a[i]
    filename = filepaths[indices[i]]
    store[filename] = prediction

# Prepare data for time series analysis, aggregating cluster counts by date
df = {}
for i in data:
    temp = [0]*K
    date_store = {}
    for j in range(0, len(data[i]), 2):
        name = data[i][j]
        date = data[i][j+1][:10]
        key = "/fs/clip-projects/m3i/data/taiwan_images/" + name
        if key in store:
            if date not in date_store:
                date_store[date] = [0]*K
            date_store[date][store[key]] += 1

    # Normalize cluster counts by date
    for j in date_store:
        sum_j = sum(date_store[j])
        date_store[j] = [xx/sum_j for xx in date_store[j]]
        df[f'{j}---{i}'] = date_store[j]

# Save the prepared data for further analysis
with open('taiwan-time-series.json', 'w') as fp:
    json.dump(df, fp)

# Plot the time series for the Taiwanese campaign
hk_plot = {}
for i in df:
    date, user = i.split('---')
    date = datetime(int(date[:4]), int(date[5:7]), int(date[8:10]))
    if date not in hk_plot:
        hk_plot[date] = 0
    hk_plot[date] += 1

# Sort and plot data
x_hk_plot, y_hk_plot = zip(*sorted(hk_plot.items()))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=90))
plt.plot(x_hk_plot, y_hk_plot)
plt.gcf().autofmt_xdate()
plt.savefig('taiwan-user-day-graph.png')
plt.clf()

# Prepare and plot time series data for another campaign for comparison
campaign_name = 'china_052020'
campaign_path = f'/fs/clip-projects/m3i/campaigns/{campaign_name}/'
references = [f'{campaign_path}china_052020_tweets_csv_unhashed.csv']

# Load reference data for the comparative campaign
collect = [pd.read_csv(i) for i in references]
reference = pd.concat(collect, ignore_index=True)
reference_store = {str(i[0]): i[13][:10] for i in reference.values}

# Load and predict clusters for embeddings of the comparative campaign
embeddings_campaign = []
indices_campaign = []
for i in tqdm(os.listdir(f'{campaign_path}embeddings')):
    a = np.load(os.path.join(f'{campaign_path}embeddings', i))
    indices_campaign.append(int(i.split('.')[0]))
    embeddings_campaign.append(a[0])

a_campaign = kmeans.predict(embeddings_campaign)

# Aggregate and normalize cluster counts for the comparative campaign
store_campaign = {}
for i, prediction in enumerate(a_campaign):
    filename = filepaths_campaign[indices_campaign[i]]
    user = filename.split('/')[-2]
    tweetid = filename.split('/')[-1].split('-')[0]
    if tweetid in reference_store:
        date_user_key = f"{reference_store[tweetid]}---{user}"
        if date_user_key not in store_campaign:
            store_campaign[date_user_key] = [0]*K
        store_campaign[date_user_key][prediction] += 1

for key in store_campaign:
    sum_values = sum(store_campaign[key])
    store_campaign[key] = [j/sum_values for j in store_campaign[key]]

# Compare and plot day-wise data between the two campaigns
user_day = {}
for key in store_campaign:
    date = datetime(int(key[:4]), int(key[5:7]), int(key[8:10]))
    user_day.setdefault(date, 0)
    user_day[date] += 1

x_user_plot, y_user_plot = zip(*sorted(user_day.items()))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=150))
plt.plot(x_user_plot, y_user_plot)
plt.gcf().autofmt_xdate()
plt.savefig(f'{campaign_name}-user-day-graph.png')
plt.clf()

# Calculate and plot average daily similarity between the campaigns
plot_values = {}
for taiwan_date in df:
    for campaign_date in store_campaign:
        if taiwan_date[:10] == campaign_date[:10]:
            temp_date = datetime(int(taiwan_date[:4]), int(taiwan_date[5:7]), int(taiwan_date[8:10]))
            cosine_similarity = 1 - cosine(df[taiwan_date], store_campaign[campaign_date])
            plot_values.setdefault(temp_date, []).append(cosine_similarity)

# Sort and plot the similarity data
x_plot, y_plot = zip(*sorted((date, sum(values)/len(values)) for date, values in plot_values.items()))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=12))
plt.plot(x_plot, y_plot)
plt.gcf().autofmt_xdate()
plt.savefig(f'Taiwan-{campaign_name}-day-graph.png')
