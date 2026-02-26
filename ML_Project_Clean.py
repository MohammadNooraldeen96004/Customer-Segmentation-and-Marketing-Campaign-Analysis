# Customer Segmentation and Marketing Campaign Analysis
# Authors: Mohammed Nouraldeen Mohammad Mohammad

### --- Imports and Initial Setup --- ###



### --- Code Sections from Notebook --- ###

import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

data = pd.read_csv(r'C:\Users\mohammad alsarese\Downloads\marketing_campaign (1).csv',sep='\t', engine='python',on_bad_lines='skip')
print("Number of datapoints:", len(data))
data.tail()

data.info()

data = data.dropna()
print("The total number of data-points after removing the rows wih missing values are:", len(data))

data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True, errors='coerce')

#Drop the null value
data = data.dropna(subset=["Dt_Customer"])

#extract the date 
dates = data["Dt_Customer"].dt.date

print("The newest customer's enrolment date in the records:", max(dates))
print("The oldest customer's enrolment date in the records:", min(dates))

#convert to  datetime  صيغة 
days = []
d1 = max(dates) 
for i in dates:
    delta = d1 - i
    days.append(delta.days)
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")
data.tail()

data["Marital_Status"].value_counts()

data["Education"].value_counts()

data["Age"] = 2021-data["Year_Birth"]

#Total spendings on various items
data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

#Deriving living situation by marital status"Alone"
data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

#Feature indicating total children living in the household
data["Children"]=data["Kidhome"]+data["Teenhome"]

#Feature for total members in the householde
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]

#Feature pertaining parenthood
data["Is_Parent"] = np.where(data.Children> 0, 1, 0)

#Segmenting education levels in three groups
data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

#For clarity
data=data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

#Dropping some of the redundant features
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
data = data.drop(to_drop, axis=1)

#Check if there are any Null values
data.isnull().sum()

#Data exploring
#To plot some selected features 
#Setting up colors prefrences
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
#Plotting following features
To_Plot = [ "Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(data[To_Plot], hue= "Is_Parent",palette= (["#682F2F","#F3AB60"]))

plt.show()

data = data[(data["Age"]<90)]
data = data[(data["Income"]<600000)]
print("The total number of data-points after removing the outliers are:", len(data))

#Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)

LE=LabelEncoder()
for i in object_cols:
    data[i]=data[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")

ds = data.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
ds = ds.drop(cols_del, axis=1)
#Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
print("All features are now scaled")

print("Dataframe to be used for further modelling:")
scaled_ds.head()

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# 1. Dendrogram
plt.figure(figsize=(12, 6))
Z = linkage(scaled_ds, method='ward')
dendrogram(Z, truncate_mode='lastp', p=12, show_leaf_counts=True)
plt.axhline(y=3.5, color='r', linestyle='--')
plt.title('Dendrogram for Cluster Determination')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# 2. Apply Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
cluster_labels = agg_clustering.fit_predict(scaled_ds)


# 3. Add cluster labels
data['Cluster_AGG'] = cluster_labels

# 4. Evaluate clustering
print("\nAgglomerative Clustering Results:")
print(f"- Number of clusters: 4")
print(f"- Points per cluster: {np.bincount(cluster_labels)}")
print(f"- Silhouette Score: {silhouette_score(scaled_ds, cluster_labels):.3f}")

# 5. PCA visualization
pca = PCA(n_components=2)  
X_pca = pca.fit_transform(scaled_ds)
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='rainbow', s=50, alpha=0.7)
plt.title('Agglomerative Clustering Results (4 Clusters)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True)
plt.show()
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Explained Variance:", pca.explained_variance_ratio_.sum())

# 6. Cluster characteristics
numerical_features = scaled_ds.columns if hasattr(scaled_ds, 'columns') else data.select_dtypes(include=np.number).columns
cluster_means = data.groupby('Cluster_AGG')[numerical_features].mean()
print("\nCluster Characteristics (mean values):")
print(cluster_means)

# 7. Bar plot
cluster_means.T.plot(kind='bar', figsize=(15, 6))
plt.title('Agglomerative Cluster Characteristics')
plt.ylabel('Mean Standardized Value')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

data.head(50)

plt.figure(figsize=(10,6))
new_sizes = data['Cluster_AGG'].value_counts()
new_sizes.plot(kind='bar', color='skyblue')
plt.title('Distribution of groups after modification')
plt.xlabel('Number of Group')
plt.ylabel('Number of element')
plt.show()

children_cols = ['Kidhome', 'Teenhome', 'Children']
avg_children = data.groupby('Cluster_AGG')[children_cols].mean()
avg_children.plot(kind='bar', figsize=(10, 6))
plt.title('Average Number of Children by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Count')
plt.legend(title='Metric')
plt.grid(True)
plt.tight_layout()
plt.show()

spending_cols = ['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']
avg_spending = data.groupby('Cluster_AGG')[spending_cols].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(avg_spending, annot=True, fmt=".1f", cmap='YlGnBu', linewidths=0.5)
plt.title("Average Spending per Category by Cluster")
plt.xlabel("Spending Category")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
if 'Income' in data.columns and 'Spent' in data.columns and 'Cluster_AGG' in data.columns:
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x='Income',
        y='Spent',
        hue='Cluster_AGG', 
        data=data,
        palette='viridis', 
        s=100,
        alpha=0.7 
    )
    plt.title('Distribution of customers by income and expenditure, colored by groups')
    plt.xlabel('Income')
    plt.ylabel('Spent')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Number of Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout() 
    plt.show()