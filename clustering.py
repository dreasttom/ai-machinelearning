import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_data)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler

#download data from
#https://www.kaggle.com/code/sylwiakmiec/clustering-kmeans-kmedoids-hc-fuzzy-dbscan-optics/data?select=Data_Cortex_Nuclear.csv

#read in data
data = pd.read_data('micedata/Data_Cortex_Nuclear.csv')
df = data.copy()
pd.set_option('display.max_row',df.shape[0])
pd.set_option('display.max_column',df.shape[1]) 
df.head()

df.shape

#review stats for specific variables
df['pCFOS_N'].describe()
df['ELK_N'].describe()
df['Bcatenin_N'].describe()


print(df['Genotype'].value_counts())

#Each of the following exports a chart for the specific data
plt.figure(figsize=(8, 4)) 
ax = sns.countplot(x='Genotype',data=df, palette='BrBG')
ax.set_title('Distribution of Genotype', fontsize=18, pad=20)
plt.show()

print(df['Treatment'].value_counts())

plt.figure(figsize=(8, 4)) 
ax = sns.countplot(x='Treatment',data=df, palette='BrBG')
ax.set_title('Distribution of Treatment', fontsize=18, pad=20)
plt.show()

print(df['Behavior'].value_counts())

plt.figure(figsize=(8, 4)) 
ax = sns.countplot(x='Behavior',data=df, palette='BrBG')
ax.set_title('Distribution of Behavior', fontsize=18, pad=20)
plt.show()

# now apply the K Nearest Neighbors algorithm
scaler = MinMaxScaler()

X = df.drop('class',axis=1)
X = pd.get_dummies(data=X,columns=["Treatment","Behavior", "Genotype"])
df_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(df_scaled)
df_scaled.head()
pca = PCA(n_components=2, random_state=42)
df_pca = pca.fit(df_scaled).transform(df_scaled)

kmeans_3 = KMeans(n_clusters=8, random_state=42)
kmeans_3 = kmeans_2.fit(df_pca)
inertia_3 = kmeans_3.inertia_
print('The clusters are:  ', kmeans_3.labels_)
print('The Inertia is:   ', kmeans_3.inertia_ )

neigh = NearestNeighbors(n_neighbors = 2)
nbrs = neigh.fit(df_pca)
distances, indices = nbrs.kneighbors(df_pca)
print(distances, indices)
