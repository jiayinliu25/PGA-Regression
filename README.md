# PGA-Regression
## Introduction
In this project, we aim to demonstrate the power of machine learning by using Python to predict the world rankings of 196 professional golf tour players based on performance data. For our analysis, we've gathered data from 2017 on 70 key metrics, including player ages, driving distances, fairway hit rates, total stroke gained, and green regulation statistics. Our initial step in this project involves constructing histograms for each dataset row, using the Python plotting library, Matplotlib. This process helps us gain a deeper understanding of the data, uncover patterns, and identify potential outliers.
'''
for c in d.columns:
    if c == "Player" or c == "COUNTRY":
        continue
    df = pd.DataFrame(d[c])
    try:
        df.hist(figsize=(7,3),edgecolor='blue')
    except:
        print(c,"here")
plt.title(c)
'''
We categorized our dataset into two distinct groups: physical data (including launch angle, driving distance, and average swing speed) and performance data (involving stroke gained, sand saves, and fairway hits). Our primary objective was to determine whether any of the physical data attributes exhibit a direct correlation with a player's points. To achieve this, we performed Pearson correlation calculations between each performance data attribute and the player's points.
'''
physical_columns = ("SHORTEST_CARRY_DISTANCE","LONGEST_CARRY_DISTANCE","AVG_CARRY_DISTANCE",
                 "SHORTEST_ACT.HANG_TIME","SHORTEST_ACT.HANG_TIME","LONGEST_ACT.HANG_TIME",
                "AVG_HANG_TIME","LOWEST_SPIN_RATE","HIGHEST_SPIN_RATE","AVG_SPIN_RATE",
                "STEEPEST_LAUNCH_ANGLE","LOWEST_LAUNCH_ANGLE","AVG_LAUNCH_ANGLE",
                "LOWEST_SF","HIGHEST_SF","AVG_SMASH_FACTOR","SLOWEST_BALL_SPEED",
                "FASTEST_BALL_SPEED","AVG_BALL_SPEED","SLOWEST_CH_SPEED","FASTEST_CH_SPEED",
                "AVG_CLUB_HEAD_SPEED","DRIVES_320+%","RTP-GOING_FOR_THE_GREEN","RTP-NOT_GOING_FOR_THE_GRN","GOING_FOR_GREEN_IN_2%","ATTEMPTS_GFG","NON-ATTEMPTS_GFG","AVG_Driving_DISTANCE")
'''               
To ensure that our analysis was consistent and meaningful, we normalized the data using a normal distribution. This was necessary because the data in these categories varied not only in their nature but also in terms of units of measurement. Normalizing the data allowed us to make accurate comparisons and identify any significant correlations between the physical attributes and player points.
### Normalize data and calculate the Pearson correlation:
'''
import scipy.stats
pairs = []
for col in physical_columns:
    pearson_co = scipy.stats.pearsonr(points, df[col])[0]
pairs.append([col, pearson_co])

pairs.sort(key=lambda x:x[1])

for col, co in pairs:
    print(col, co)
'''    
From the Pearson Correlation calculated, most of them do not have any strong correlation. However, one interesting correlation to note is the slightly strong positive correlation in average driving distance, average carry distance, longest carry distance and the number of drives above 320 yards(all of them ranging between 0.40 to 0.47). Furthermore, there is also a slightly strong negative correlation in RTP-going for the green(the total of the score gained (-1 or -2 for example) on the hole relative to par, a player is assumed to be going for the green on the first shot on a par 4 and second shot on a par 5).

In addition, players’ points distribution is also visualized using k-means clustering: 
### One Variable 
'''
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
'''
