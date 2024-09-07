## Ivan Suarez
## Dataset: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
## Lookerstudio link: https://lookerstudio.google.com/reporting/619e0a2f-3a9a-4c29-b9f9-ea8cd744715d 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Data Visualization 
import seaborn as sns  #Python library for Visualization
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Import the dataset
dataset = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

#Exploratory Data Analysis
#As this is unsupervised learning so Label (Output Column) is unknown
print(dataset.head(10)) #Printing first 10 rows of the dataset

#total rows and colums in the dataset
print(dataset.shape)

print(dataset.info()) # there are no missing values as all the columns has 200 entries properly

#Missing values computation
print(dataset.isnull().sum())

sns.histplot(dataset['Occupation'], bins=10, kde=True)
plt.show()

### Feature sleection for the model
#Considering only 3 features (Age, Sleep Duration, Quality of Sleep,Physical Activity Level) and no Label available
X = dataset.iloc[:, [2,4,5,6]].values

#Building the Model
wcss=[]

#we always assume the max number of cluster would be 10
#you can judge the number of clusters by doing averaging
###Static code to get max no of clusters

clusters = 10

for i in range(1, clusters + 1):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

    #inertia_ is the formula used to segregate the data points into clusters

#Visualizing the ELBOW method to get the optimal value of K 
plt.plot(range(1, clusters + 1), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

#Model Build
kmeansmodel = KMeans(n_clusters= clusters, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)

#Visualizing all the clusters 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of professionals')
plt.xlabel('Behavior Score')
plt.ylabel('Quality of Sleep')
plt.legend()
plt.show()