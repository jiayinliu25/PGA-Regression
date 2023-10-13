# PGA-Regression
## Introduction
In this project, we aim to demonstrate the power of machine learning by using Python to predict the world rankings of 196 professional golf tour players based on performance data. For our analysis, we've gathered data from 2017 on 70 key metrics, including player ages, driving distances, fairway hit rates, total stroke gained, and green regulation statistics. Our initial step in this project involves constructing histograms for each dataset row, using the Python plotting library, Matplotlib. This process helps us gain a deeper understanding of the data, uncover patterns, and identify potential outliers.
### The Data: 
```
for c in d.columns:
    if c == "Player" or c == "COUNTRY":
        continue
    df = pd.DataFrame(d[c])
    try:
        df.hist(figsize=(7,3),edgecolor='blue')
    except:
        print(c,"here")
plt.title(c)
```
We categorized our dataset into two distinct groups: physical data (including launch angle, driving distance, and average swing speed) and performance data (involving stroke gained, sand saves, and fairway hits). Our primary objective was to determine whether any of the physical data attributes exhibit a direct correlation with a player's points. To achieve this, we performed Pearson correlation calculations between each performance data attribute and the player's points.
```
physical_columns = ("SHORTEST_CARRY_DISTANCE","LONGEST_CARRY_DISTANCE","AVG_CARRY_DISTANCE",
                 "SHORTEST_ACT.HANG_TIME","SHORTEST_ACT.HANG_TIME","LONGEST_ACT.HANG_TIME",
                "AVG_HANG_TIME","LOWEST_SPIN_RATE","HIGHEST_SPIN_RATE","AVG_SPIN_RATE",
                "STEEPEST_LAUNCH_ANGLE","LOWEST_LAUNCH_ANGLE","AVG_LAUNCH_ANGLE",
                "LOWEST_SF","HIGHEST_SF","AVG_SMASH_FACTOR","SLOWEST_BALL_SPEED",
                "FASTEST_BALL_SPEED","AVG_BALL_SPEED","SLOWEST_CH_SPEED","FASTEST_CH_SPEED",
                "AVG_CLUB_HEAD_SPEED","DRIVES_320+%","RTP-GOING_FOR_THE_GREEN","RTP-NOT_GOING_FOR_THE_GRN","GOING_FOR_GREEN_IN_2%","ATTEMPTS_GFG","NON-ATTEMPTS_GFG","AVG_Driving_DISTANCE")
```           
To ensure that our analysis was consistent and meaningful, we normalized the data using a normal distribution. This was necessary because the data in these categories varied not only in their nature but also in terms of units of measurement. Normalizing the data allowed us to make accurate comparisons and identify any significant correlations between the physical attributes and player points.
### Normalize and Calculate the Pearson Correlation:
```
import scipy.stats
pairs = []
for col in physical_columns:
    pearson_co = scipy.stats.pearsonr(points, df[col])[0]
pairs.append([col, pearson_co])

pairs.sort(key=lambda x:x[1])

for col, co in pairs:
    print(col, co)
```  
From the Pearson Correlation calculated, most of them do not have any strong correlation. However, one interesting correlation to note is the slightly strong positive correlation in average driving distance, average carry distance, longest carry distance and the number of drives above 320 yards(all of them ranging between 0.40 to 0.47). Furthermore, there is also a slightly strong negative correlation in RTP-going for the green(the total of the score gained (-1 or -2 for example) on the hole relative to par, a player is assumed to be going for the green on the first shot on a par 4 and second shot on a par 5).


### One Variable K-Means Clustering:
In addition, players’ points distribution is also visualized using k-means clustering: 
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.figure(figsize=(15,1))
d = pd.read_csv('PGATOUR_meta2.csv',delimiter = ',')
xp = list(d.POINTS)
df= pd.DataFrame(xp)
y = [0] * len(xp)
X = list(zip(y, xp))
k_means = KMeans( n_clusters=5)
k_means.fit(X)
plt.scatter(xp, np.zeros_like(y), c=k_means.labels_, cmap='rainbow', s= 5)
plt.scatter(k_means.cluster_centers_[:,1] ,k_means.cluster_centers_[:,0],  color='black')
plt.axis([0, 6000, -1, 1])
plt.show
```
### k-means Clustering Using Scikit Learn:
```
import sklearn.preprocessing
physical_data = df[list(physical_columns)]
norm_physical_data = physical_data.copy()
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(norm_physical_data)

"""
for col in physical_columns:
    norm_physical_data[col] = sklearn.preprocessing.norm_physical_data[col]
"""

norm_physical_data = scaler.transform(norm_physical_data)

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)
k_means.fit(norm_physical_data)

plt.scatter(k_means.labels_, df["POINTS"])
plt.show()
```
## Conclusion & Limitations 
In summary, while this project didn't yield a precise prediction of players' future rankings, it's essential to recognize the inherent complexity of golf, a sport with numerous unpredictable variables. Although our analysis mainly focused on players' physical data, which can change from year to year, we uncovered some intriguing insights. Notably, we identified that driving distance and the frequency of going for the green relative to par show relatively strong correlations with players' points.

These findings suggest a positive link between driving distance and player points, and a negative correlation between the frequency of going for the green (measured in negative numbers) and player points. This insight provides valuable direction for further investigation and analysis.

In my view, the process of analyzing and visualizing existing data is an important step, and while creating predictive models remains a challenge, it represents an exciting opportunity for future improvement. I'm eager to enhance this project by incorporating more data and refining our analytical techniques to make more accurate predictions down the line.
