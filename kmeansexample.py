import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as seab 
#you will need https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
import os
import warnings

# KMeans Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# it is likely you won't have seaborn
# just use pip install seaborn
import seaborn as seab
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

#read in the data, make sure you have downloaded it from
# https://www.kaggle.com/code/pragyamukherjee/alzheimer-s-k-means/data
data=pd.read_csv("alzheimerdata/oasis_cross-sectional.csv")
data.head()
data.Hand.unique()
data.drop(["ID","Hand","Delay"],axis=1,inplace=True)

data.columns=["gender","age","education","soc_eco_st","mini_mental_state_exam","clinical_dementia_rating",
           "estimated_total_intracranial_volume","normalize_whole_brain_volume","atlas_scaling_factor"]


data.gender = [1 if each == "F" else 0 for each in  data.gender]

data.info()

data.isnull().sum()
data.describe()
def impute_median(series):
    return series.fillna(series.median())

data.education =data["education"].transform(impute_median)
data.soc_eco_st =data["soc_eco_st"].transform(impute_median)
data.mini_mental_state_exam =data["mini_mental_state_exam"].transform(impute_median)
data.clinical_dementia_rating =data["clinical_dementia_rating"].transform(impute_median)

#visualize the correlation
plot.figure(figsize=(15,10))
seab.heatmap(data.iloc[:,0:10].corr(), annot=True,fmt=".0%")
plot.show()


kmeans = KMeans(
init="random",
n_clusters=2,
n_init=10,
max_iter=300,
random_state=42
)
kmeans.fit(data.loc[:,['age','normalize_whole_brain_volume']])



pred = kmeans.predict(data.loc[:,['age','normalize_whole_brain_volume']])
plot.scatter(data= data,x ='age',y = 'normalize_whole_brain_volume',c = pred, cmap='viridis')
plot.xlabel('age')
plot.ylabel('clinical_dementia_rating')
plot.show()

kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42,}
sse = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data.loc[:,['age','normalize_whole_brain_volume']])
    sse.append(kmeans.inertia_)
    
plot.style.use("fivethirtyeight")
plot.plot(range(1, 6), sse)
plot.xticks(range(1, 6))
plot.xlabel("No. of Clusters")
plot.ylabel("SSE")
# Sum of Squares (SSE) is the sum of the squared differences
# between each observation and that  group's mean.
plot.show()
