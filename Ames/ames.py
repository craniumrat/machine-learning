import mlflow
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

## We can autolog it, but I want to add the cluster result as a PNG to each run.
## mlflow.autolog()

plt.style.use('seaborn-whitegrid')
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight=14, titlepad=10)

def score_dataset(X, y, model=XGBRegressor(random_state=0)):
  # Label encoding for categoricals
  for colname in X.select_dtypes(["category", "object"]):
    X[colname], _ = X[colname].factorize()
    
  # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
  score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_log_error")

  score = -1 * score.mean()
  score = np.sqrt(score)
  return score

with ZipFile('ames.csv.zip') as zipfile:
  with zipfile.open('ames.csv') as csv_file:
    df = pd.read_csv(csv_file)
    print(df.shape)

    X = df.copy()
    y = X.pop('SalePrice')

    features = ['LotArea', 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF', 'GrLivArea']

    for cluster_count in range(2,10):
          with mlflow.start_run() as run:
            X_scaled = X.loc[:, features]
            X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)

            kmeans = KMeans(n_clusters=cluster_count, n_init='auto', random_state=0)
            X['Cluster'] = kmeans.fit_predict(X_scaled)

            ##Now generate the cluster map and add it to mlflow.
            Xy = X.copy()
            Xy['Cluster'] = Xy.Cluster.astype("category")
            Xy["SalePrice"] = y

            plot = sns.relplot(x="value", y="SalePrice", hue="Cluster", col="variable",
                       height = 4, aspect=1, facet_kws={'sharex' : False}, col_wrap=3,
                       data = Xy.melt(value_vars=features, id_vars=["SalePrice", "Cluster"]),
            )

            mlflow.sklearn.log_model(kmeans, "model")
            mlflow.log_param("n_clusters", cluster_count)
            filename = 'plots/clusters_' + str(cluster_count) + '.png'
            plt.savefig(filename)
            mlflow.log_artifact(filename)



          ## This is for later.
          ## print("n_clusters: " + str(cluster_count) + ". Score: " + str(score_dataset(X, y)))


