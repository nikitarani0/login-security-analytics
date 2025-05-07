import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
logins = []
start_time = datetime.now()

for i in range(1000):
    timestamp = start_time + timedelta(seconds= i * random.randint(1,5))
    user_id = random.choice(['user1', 'user2', 'user3', 'admin' , 'guest'])
    success = np.random.choice([1,0], p= [0.95,0.05])
    ip_address = f"192.168.0.{random.randint(1, 255)}"
    logins.append([timestamp, user_id, success, ip_address])

logs_df = pd.DataFrame(logins, columns=['timestamp', 'user', 'success', 'ip'])
logs_df.to_csv("system_logs.csv", index=False)

data = pd.read_csv('system_logs.csv')
# Convert timestamp to numeric for analysis
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['timestamp_int'] = data['timestamp'].astype("int64") // 1e9 #seconds
data['success'] = data['success'].astype('int')

# Encode categorical data
data['user_encoded'] = data['user'].astype('category').cat.codes

from sklearn.ensemble import IsolationForest
features = data[['timestamp_int', 'success', 'user_encoded']]

model = IsolationForest(contamination=0.05, random_state=42)
data['anomaly'] = model.fit_predict(features)

# -1 means anomaly, 1 means normal
data['anomaly'] = data['anomaly'].map({1:0 , -1:1})

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
sns.scatterplot(data=data, x='timestamp', y='user', hue='anomaly', palette={0: 'green', 1: 'red'})
plt.title("Anomaly detect in login log files")
plt.xlabel("timestamp")
plt.ylabel("user")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()
