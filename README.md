<h3 align='center'>Classify Song Genres from Audio Data<br>Rock or rap?</h3>

<!-- <p align='center'>
  <img width=500 height=300 src='https://i.ytimg.com/vi/oPgWYj2smCw/maxresdefault.jpg'>
</p> -->

<h3>1. Preparing our dataset</h3>
<p>    Over the past few years, streaming services with huge catalogs have become the primary means through which most people listen to their favorite music. For this reason, streaming services have looked into means of categorizing music to allow for personalized recommendations.<br><br>Let's load the metadata about our tracks alongside the track metrics compiled by <b>The Echo Nest</b>.</p>

```python
import pandas as pd

tracks = pd.read_csv('datasets/fma-rock-vs-hiphop.csv')
echonest_metrics = pd.read_json('datasets/echonest-metrics.json', precise_float=True)

# Merge the relevant columns of tracks and echonest_metrics
echo_tracks = pd.merge(echonest_metrics, tracks[['track_id' , 'genre_top']], how='inner', on='track_id')

# Inspect the resultant dataframe
echo_tracks.info()
```
|   Int64Index: 4802 entries, 0 to 4801   |                       |
|:---------------------------------------:|:---------------------:|
|     Data columns (total 10 columns):    |                       |
|               acousticness              | 4802 non-null float64 |
|               danceability              | 4802 non-null float64 |
|                  energy                 | 4802 non-null float64 |
|             instrumentalness            | 4802 non-null float64 |
|                 liveness                | 4802 non-null float64 |
|               speechiness               | 4802 non-null float64 |
|                  tempo                  | 4802 non-null float64 |
|                 track_id                | 4802 non-null int64   |
|                 valence                 | 4802 non-null float64 |
|                genre_top                | 4802 non-null object  |
| dtypes: float64(8), int64(1), object(1) |                       |
|         memory usage: 412.7+ KB         |                       |


<h3>2. Pairwise relationships between continuous variables</h3>
<p>We want to avoid using variables that have strong correlations with each other -- hence avoiding feature redundancy<br>To get a sense of whether there are any strongly correlated features in our data, we will use built-in functions in the <code>pandas</code> package <code>.corr()</code>. </p>

```python
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()
```
<p align='center'>
  <img src='datasets/corr.jpg'>
</p>

<h3>3. Normalizing the feature data</h3>
<p>Since we didn't find any particular strong correlations between our features, we can instead use a common approach to reduce the number of features called <b>principal component analysis (PCA)</b><br>To avoid bias, I first normalize the data using <code>sklearn</code> built-in <code>StandardScaler</code> method</p>

```python
from sklearn.preprocessing import StandardScaler

features = echo_tracks.drop(['track_id', 'genre_top'], axis=1)
labels = echo_tracks.genre_top

scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(features)
```

<h3>4. Principal Component Analysis on our scaled data</h3>
<p>Now PCA is ready to determine by how much we can reduce the dimensionality of our data. We can use <b>scree-plots</b> and <b>cumulative explained ratio plots</b> to find the number of components to use in further analyses.</p>
  
```python
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(scaled_train_features)

exp_variance = pca.explained_variance_ratio_

fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component #')
```

<img src='datasets/PCAhist.jpg'>





<h3>5. Further visualization of PCA</h3>





<h3>6. Train a decision tree to classify genre</h3>











<h3>7. Compare our decision tree to a logistic regression</h3>










<h3>8. Balance our data for greater performance</h3>










<h3>9. Does balancing our dataset improve model bias?</h3>







<h3>10. Using cross-validation to evaluate our models</h3>
