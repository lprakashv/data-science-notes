# Recommendation

Purpose: to find and recommend items that a user is most likely to be interested in.

### Examples of Recommendation Engines:

1. Product: Amazon, Etsy
2. Movie: Netflix
3. Music: Apple Music, Spotify etc
4. Social connections: Facebook, Linkedin, Instagram

## Simple Appoaches to Recommender Systems:

### 1. POPULARITY-BASED RECOMMENDERS

**Based on simple copunt statistics** (numer of ratings given to an item)

| user   | place   | rating |
| :----:   | :----:    | :----:   |
| user A | place 1 | 10     |
| user B | place 1 | 8 |
| user C | place 2 | 8 |
| user D | place 2 | 7 |
| user E | place 1 | 8 |
| user F | place 1 | 7 |
| user G | place 1 | 10 |
| | ![](https://cdnjs.cloudflare.com/ajax/libs/fontisto/3.0.4/icons/directional/arrow-down.png) | |

| place   | rating count |
|:-------:|:------------:|
| place 1 | 5 |
| place 2 | 2 |

Fun facts on **Popularity based recommenders**:
- rely on purchase history data
- are often used by online news sites like Bloomberg
- cannot produce personalized result


```python
import pandas as pd
import numpy as np
```


```python
ratings = pd.read_csv('rating_final.csv')
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userID</th>
      <th>placeID</th>
      <th>rating</th>
      <th>food_rating</th>
      <th>service_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>U1077</td>
      <td>135085</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>U1077</td>
      <td>135038</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U1077</td>
      <td>132825</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U1077</td>
      <td>135060</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>U1068</td>
      <td>135104</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
cuisines = pd.read_csv('chefmozcuisine.csv')
cuisines.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>placeID</th>
      <th>Rcuisine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>135110</td>
      <td>Spanish</td>
    </tr>
    <tr>
      <th>1</th>
      <td>135109</td>
      <td>Italian</td>
    </tr>
    <tr>
      <th>2</th>
      <td>135107</td>
      <td>Latin_American</td>
    </tr>
    <tr>
      <th>3</th>
      <td>135106</td>
      <td>Mexican</td>
    </tr>
    <tr>
      <th>4</th>
      <td>135105</td>
      <td>Fast_Food</td>
    </tr>
  </tbody>
</table>
</div>




```python
# how many people have given ratings to a particuar 'place'
ratings_counts = pd.DataFrame(ratings.groupby('placeID')['rating'].count())
ratings_counts.sort_values('rating', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135085</th>
      <td>36</td>
    </tr>
    <tr>
      <th>132825</th>
      <td>32</td>
    </tr>
    <tr>
      <th>135032</th>
      <td>28</td>
    </tr>
    <tr>
      <th>135052</th>
      <td>25</td>
    </tr>
    <tr>
      <th>132834</th>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>




```python
cuisine_popularity = pd.merge(ratings_count, cuisines, on='placeID')
cuisine_popularity.sort_values('rating', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>placeID</th>
      <th>rating</th>
      <th>Rcuisine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>105</th>
      <td>135085</td>
      <td>36</td>
      <td>Fast_Food</td>
    </tr>
    <tr>
      <th>28</th>
      <td>132825</td>
      <td>32</td>
      <td>Mexican</td>
    </tr>
    <tr>
      <th>71</th>
      <td>135032</td>
      <td>28</td>
      <td>Cafeteria</td>
    </tr>
    <tr>
      <th>72</th>
      <td>135032</td>
      <td>28</td>
      <td>Contemporary</td>
    </tr>
    <tr>
      <th>86</th>
      <td>135052</td>
      <td>25</td>
      <td>Bar_Pub_Brewery</td>
    </tr>
  </tbody>
</table>
</div>




```python
cuisine_popularity['Rcuisine'].describe()
```




    count         112
    unique         23
    top       Mexican
    freq           28
    Name: Rcuisine, dtype: object




```python
cuisines['Rcuisine'].describe()
```




    count         916
    unique         59
    top       Mexican
    freq          239
    Name: Rcuisine, dtype: object



### 2. CORRELATION BASED RECOMMENDER SYSTEMS

**Pearson's correlation coefficient (r) - "Pearson's r"**

|       r  | description  |
|----------|--------------|
| *r = 1*  | Strong positive *linear* relationship |
| *r = 0*  | Not linearly correlated               |
| *r = -1* | Strong negative *linear* relationship |

#### Item based similarity:
    Recommend an item based on how well it correlates with other items with respect to user ratings


```python
ratings = pd.read_csv('rating_final.csv')
cuisines = pd.read_csv('chefmozcuisine.csv')
geodata = pd.read_csv('geoplaces2.csv')
```


```python
geodata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>placeID</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>the_geom_meter</th>
      <th>name</th>
      <th>address</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>fax</th>
      <th>...</th>
      <th>alcohol</th>
      <th>smoking_area</th>
      <th>dress_code</th>
      <th>accessibility</th>
      <th>price</th>
      <th>url</th>
      <th>Rambience</th>
      <th>franchise</th>
      <th>area</th>
      <th>other_services</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>134999</td>
      <td>18.915421</td>
      <td>-99.184871</td>
      <td>0101000020957F000088568DE356715AC138C0A525FC46...</td>
      <td>Kiku Cuernavaca</td>
      <td>Revolucion</td>
      <td>Cuernavaca</td>
      <td>Morelos</td>
      <td>Mexico</td>
      <td>?</td>
      <td>...</td>
      <td>No_Alcohol_Served</td>
      <td>none</td>
      <td>informal</td>
      <td>no_accessibility</td>
      <td>medium</td>
      <td>kikucuernavaca.com.mx</td>
      <td>familiar</td>
      <td>f</td>
      <td>closed</td>
      <td>none</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132825</td>
      <td>22.147392</td>
      <td>-100.983092</td>
      <td>0101000020957F00001AD016568C4858C1243261274BA5...</td>
      <td>puesto de tacos</td>
      <td>esquina santos degollado y leon guzman</td>
      <td>s.l.p.</td>
      <td>s.l.p.</td>
      <td>mexico</td>
      <td>?</td>
      <td>...</td>
      <td>No_Alcohol_Served</td>
      <td>none</td>
      <td>informal</td>
      <td>completely</td>
      <td>low</td>
      <td>?</td>
      <td>familiar</td>
      <td>f</td>
      <td>open</td>
      <td>none</td>
    </tr>
    <tr>
      <th>2</th>
      <td>135106</td>
      <td>22.149709</td>
      <td>-100.976093</td>
      <td>0101000020957F0000649D6F21634858C119AE9BF528A3...</td>
      <td>El Rinc�n de San Francisco</td>
      <td>Universidad 169</td>
      <td>San Luis Potosi</td>
      <td>San Luis Potosi</td>
      <td>Mexico</td>
      <td>?</td>
      <td>...</td>
      <td>Wine-Beer</td>
      <td>only at bar</td>
      <td>informal</td>
      <td>partially</td>
      <td>medium</td>
      <td>?</td>
      <td>familiar</td>
      <td>f</td>
      <td>open</td>
      <td>none</td>
    </tr>
    <tr>
      <th>3</th>
      <td>132667</td>
      <td>23.752697</td>
      <td>-99.163359</td>
      <td>0101000020957F00005D67BCDDED8157C1222A2DC8D84D...</td>
      <td>little pizza Emilio Portes Gil</td>
      <td>calle emilio portes gil</td>
      <td>victoria</td>
      <td>tamaulipas</td>
      <td>?</td>
      <td>?</td>
      <td>...</td>
      <td>No_Alcohol_Served</td>
      <td>none</td>
      <td>informal</td>
      <td>completely</td>
      <td>low</td>
      <td>?</td>
      <td>familiar</td>
      <td>t</td>
      <td>closed</td>
      <td>none</td>
    </tr>
    <tr>
      <th>4</th>
      <td>132613</td>
      <td>23.752903</td>
      <td>-99.165076</td>
      <td>0101000020957F00008EBA2D06DC8157C194E03B7B504E...</td>
      <td>carnitas_mata</td>
      <td>lic. Emilio portes gil</td>
      <td>victoria</td>
      <td>Tamaulipas</td>
      <td>Mexico</td>
      <td>?</td>
      <td>...</td>
      <td>No_Alcohol_Served</td>
      <td>permitted</td>
      <td>informal</td>
      <td>completely</td>
      <td>medium</td>
      <td>?</td>
      <td>familiar</td>
      <td>t</td>
      <td>closed</td>
      <td>none</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
places = geodata[['placeID', 'name']]
places.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>placeID</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>134999</td>
      <td>Kiku Cuernavaca</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132825</td>
      <td>puesto de tacos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>135106</td>
      <td>El Rinc�n de San Francisco</td>
    </tr>
    <tr>
      <th>3</th>
      <td>132667</td>
      <td>little pizza Emilio Portes Gil</td>
    </tr>
    <tr>
      <th>4</th>
      <td>132613</td>
      <td>carnitas_mata</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Average Rating based ranking w.r.t. places
place_avg_ratings = pd.DataFrame(ratings.groupby('placeID')['rating'].mean())
place_avg_ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132560</th>
      <td>0.50</td>
    </tr>
    <tr>
      <th>132561</th>
      <td>0.75</td>
    </tr>
    <tr>
      <th>132564</th>
      <td>1.25</td>
    </tr>
    <tr>
      <th>132572</th>
      <td>1.00</td>
    </tr>
    <tr>
      <th>132583</th>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
place_avg_ratings['rating_count'] = pd.DataFrame(ratings.groupby('placeID')['rating'].count())
place_avg_ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>rating_count</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132560</th>
      <td>0.50</td>
      <td>4</td>
    </tr>
    <tr>
      <th>132561</th>
      <td>0.75</td>
      <td>4</td>
    </tr>
    <tr>
      <th>132564</th>
      <td>1.25</td>
      <td>4</td>
    </tr>
    <tr>
      <th>132572</th>
      <td>1.00</td>
      <td>15</td>
    </tr>
    <tr>
      <th>132583</th>
      <td>1.00</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
place_avg_ratings.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>rating_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>130.000000</td>
      <td>130.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.179622</td>
      <td>8.930769</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.349354</td>
      <td>6.124279</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.250000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.181818</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.400000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000000</td>
      <td>36.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
place_avg_ratings.sort_values('rating_count', ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>rating_count</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135085</th>
      <td>1.333333</td>
      <td>36</td>
    </tr>
    <tr>
      <th>132825</th>
      <td>1.281250</td>
      <td>32</td>
    </tr>
    <tr>
      <th>135032</th>
      <td>1.178571</td>
      <td>28</td>
    </tr>
    <tr>
      <th>135052</th>
      <td>1.280000</td>
      <td>25</td>
    </tr>
    <tr>
      <th>132834</th>
      <td>1.000000</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings_cross_table = pd.pivot_table(data=ratings, values='rating', index='userID', columns='placeID')
ratings_cross_table.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>placeID</th>
      <th>132560</th>
      <th>132561</th>
      <th>132564</th>
      <th>132572</th>
      <th>132583</th>
      <th>132584</th>
      <th>132594</th>
      <th>132608</th>
      <th>132609</th>
      <th>132613</th>
      <th>...</th>
      <th>135080</th>
      <th>135081</th>
      <th>135082</th>
      <th>135085</th>
      <th>135086</th>
      <th>135088</th>
      <th>135104</th>
      <th>135106</th>
      <th>135108</th>
      <th>135109</th>
    </tr>
    <tr>
      <th>userID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>U1001</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>U1002</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>U1003</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>U1004</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>U1005</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 130 columns</p>
</div>




```python
most_popular_place = place_avg_ratings.iloc[place_avg_ratings['rating_count'].argmax()].name
print("most popular place: {}".format(most_popular_place))

most_popular_cuisine = cuisines.loc[cuisines['placeID'] == most_popular_place]['Rcuisine'].name
print("most popular cuisine: {}".format(most_popular_cuisine))

most_popular_place_ratings = ratings_cross_table[most_popular_place]
most_popular_place_ratings[most_popular_place_ratings.notnull()]
```

    most popular place: 135085
    most popular cuisine: Rcuisine





    userID
    U1001    0.0
    U1002    1.0
    U1007    1.0
    U1013    1.0
    U1016    2.0
    U1027    1.0
    U1029    1.0
    U1032    1.0
    U1033    2.0
    U1036    2.0
    U1045    2.0
    U1046    1.0
    U1049    0.0
    U1056    2.0
    U1059    2.0
    U1062    0.0
    U1077    2.0
    U1081    1.0
    U1084    2.0
    U1086    2.0
    U1089    1.0
    U1090    2.0
    U1092    0.0
    U1098    1.0
    U1104    2.0
    U1106    2.0
    U1108    1.0
    U1109    2.0
    U1113    1.0
    U1116    2.0
    U1120    0.0
    U1122    2.0
    U1132    2.0
    U1134    2.0
    U1135    0.0
    U1137    2.0
    Name: 135085, dtype: float64




```python
# Evaluating similarity based on "correlation":

similar_to_most_popular = ratings_cross_table.corrwith(most_popular_place_ratings)

corr_most_popular = pd.DataFrame(similar_to_most_popular, columns=['PearsonsR'])
corr_most_popular.dropna(inplace=True)
corr_most_popular.head()
```

    /opt/anaconda3/lib/python3.8/site-packages/numpy/lib/function_base.py:2634: RuntimeWarning: Degrees of freedom <= 0 for slice
      c = cov(x, y, rowvar, dtype=dtype)
    /opt/anaconda3/lib/python3.8/site-packages/numpy/lib/function_base.py:2493: RuntimeWarning: divide by zero encountered in true_divide
      c *= np.true_divide(1, fact)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PearsonsR</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132572</th>
      <td>-0.428571</td>
    </tr>
    <tr>
      <th>132723</th>
      <td>0.301511</td>
    </tr>
    <tr>
      <th>132754</th>
      <td>0.930261</td>
    </tr>
    <tr>
      <th>132825</th>
      <td>0.700745</td>
    </tr>
    <tr>
      <th>132834</th>
      <td>0.814823</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr_with_most_pop_ratcnt = corr_most_popular.join(place_avg_ratings['rating_count'])
corr_with_most_pop_ratcnt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PearsonsR</th>
      <th>rating_count</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132572</th>
      <td>-0.428571</td>
      <td>15</td>
    </tr>
    <tr>
      <th>132723</th>
      <td>0.301511</td>
      <td>12</td>
    </tr>
    <tr>
      <th>132754</th>
      <td>0.930261</td>
      <td>13</td>
    </tr>
    <tr>
      <th>132825</th>
      <td>0.700745</td>
      <td>32</td>
    </tr>
    <tr>
      <th>132834</th>
      <td>0.814823</td>
      <td>25</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_10_places_like_most_pop = corr_with_most_pop_ratcnt[corr_with_most_pop_ratcnt['rating_count'] >= 10
                                                       ].sort_values('PearsonsR', ascending=False
                                                                    )[:10]
top_10_places_like_most_pop
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PearsonsR</th>
      <th>rating_count</th>
    </tr>
    <tr>
      <th>placeID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135076</th>
      <td>1.000000</td>
      <td>13</td>
    </tr>
    <tr>
      <th>135085</th>
      <td>1.000000</td>
      <td>36</td>
    </tr>
    <tr>
      <th>135066</th>
      <td>1.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>132754</th>
      <td>0.930261</td>
      <td>13</td>
    </tr>
    <tr>
      <th>135045</th>
      <td>0.912871</td>
      <td>13</td>
    </tr>
    <tr>
      <th>135062</th>
      <td>0.898933</td>
      <td>21</td>
    </tr>
    <tr>
      <th>135028</th>
      <td>0.892218</td>
      <td>15</td>
    </tr>
    <tr>
      <th>135042</th>
      <td>0.881409</td>
      <td>20</td>
    </tr>
    <tr>
      <th>135046</th>
      <td>0.867722</td>
      <td>11</td>
    </tr>
    <tr>
      <th>132872</th>
      <td>0.840168</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(pd.merge(top_10_places_like_most_pop, cuisines, on='placeID'), places, on='placeID')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>placeID</th>
      <th>PearsonsR</th>
      <th>rating_count</th>
      <th>Rcuisine</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>135085</td>
      <td>1.000000</td>
      <td>36</td>
      <td>Fast_Food</td>
      <td>Tortas Locas Hipocampo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>132754</td>
      <td>0.930261</td>
      <td>13</td>
      <td>Mexican</td>
      <td>Cabana Huasteca</td>
    </tr>
    <tr>
      <th>2</th>
      <td>135028</td>
      <td>0.892218</td>
      <td>15</td>
      <td>Mexican</td>
      <td>La Virreina</td>
    </tr>
    <tr>
      <th>3</th>
      <td>135042</td>
      <td>0.881409</td>
      <td>20</td>
      <td>Chinese</td>
      <td>Restaurant Oriental Express</td>
    </tr>
    <tr>
      <th>4</th>
      <td>135046</td>
      <td>0.867722</td>
      <td>11</td>
      <td>Fast_Food</td>
      <td>Restaurante El Reyecito</td>
    </tr>
    <tr>
      <th>5</th>
      <td>132872</td>
      <td>0.840168</td>
      <td>12</td>
      <td>American</td>
      <td>Pizzeria Julios</td>
    </tr>
  </tbody>
</table>
</div>



## Collaborative Filtering Recommenders

**Recommend items based on crowdsourced information about users' preferences for items**.

2 approaches:
1. User based

    *Based on known user attributes, we know that User B is similar to User D. User D really likes his life insurance policy, so let's recomment it to Uesr B also.*

2. Item based

    *User B and User D both gave high ratings to the cell phone and the cell phone case. Since User A also likes the cell phone, let's recommend to her the cell phone case also.*
    
User attributes can be described as a list of values (possibly boolean).

### Classification-Based Collaborative Filtering

Provides personalizarion by accepting:
- user and item attribute data
- purchase history data
- other contextual data
- Gives a Yes/No classification! (Will he/she accept/purchase?)

Example classification methods:

1. Naive Bayes classification
2. Logistic regression

#### 1. Logistic Regression as classifier


```python
import numpy as np
import pandas as pd

from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
```


```python
bank_full = pd.read_csv('bank_full_w_dummy_vars.csv')
bank_full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>...</th>
      <th>job_unknown</th>
      <th>job_retired</th>
      <th>job_services</th>
      <th>job_self_employed</th>
      <th>job_unemployed</th>
      <th>job_maid</th>
      <th>job_student</th>
      <th>married</th>
      <th>single</th>
      <th>divorced</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>2143</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>technician</td>
      <td>single</td>
      <td>secondary</td>
      <td>no</td>
      <td>29</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>unknown</td>
      <td>no</td>
      <td>1506</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>unknown</td>
      <td>single</td>
      <td>unknown</td>
      <td>no</td>
      <td>1</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




```python
bank_full.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45211 entries, 0 to 45210
    Data columns (total 37 columns):
     #   Column                        Non-Null Count  Dtype 
    ---  ------                        --------------  ----- 
     0   age                           45211 non-null  int64 
     1   job                           45211 non-null  object
     2   marital                       45211 non-null  object
     3   education                     45211 non-null  object
     4   default                       45211 non-null  object
     5   balance                       45211 non-null  int64 
     6   housing                       45211 non-null  object
     7   loan                          45211 non-null  object
     8   contact                       45211 non-null  object
     9   day                           45211 non-null  int64 
     10  month                         45211 non-null  object
     11  duration                      45211 non-null  int64 
     12  campaign                      45211 non-null  int64 
     13  pdays                         45211 non-null  int64 
     14  previous                      45211 non-null  int64 
     15  poutcome                      45211 non-null  object
     16  y                             45211 non-null  object
     17  y_binary                      45211 non-null  int64 
     18  housing_loan                  45211 non-null  int64 
     19  credit_in_default             45211 non-null  int64 
     20  personal_loans                45211 non-null  int64 
     21  prev_failed_to_subscribe      45211 non-null  int64 
     22  prev_subscribed               45211 non-null  int64 
     23  job_management                45211 non-null  int64 
     24  job_tech                      45211 non-null  int64 
     25  job_entrepreneur              45211 non-null  int64 
     26  job_bluecollar                45211 non-null  int64 
     27  job_unknown                   45211 non-null  int64 
     28  job_retired                   45211 non-null  int64 
     29  job_services                  45211 non-null  int64 
     30  job_self_employed             45211 non-null  int64 
     31  job_unemployed                45211 non-null  int64 
     32  job_maid                      45211 non-null  int64 
     33  job_student                   45211 non-null  int64 
     34  married                       45211 non-null  int64 
     35  single                        45211 non-null  int64 
     36  divorced                      45211 non-null  int64 
    dtypes: int64(27), object(10)
    memory usage: 12.8+ MB



```python
X = bank_full.iloc[:,list(range(18,37))].values

y = bank_full.iloc[:,17].values

print("X shape = {}".format(X.shape))
print("y shape = {}".format(y.shape))
```

    X shape = (45211, 19)
    y shape = (45211,)



```python
log_reg = LogisticRegression()
log_reg.fit(X, y)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
new_user = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
y_pred = log_reg.predict([new_user])
y_pred
```




    array([0])



#### 2. Model-based Collaborative filtering

##### Singular Value Decomposition (SVD)
- A linear algebra method that can decompose utilituy matrix into three compressed matrices.
- Model-based recommender - use these ompressed matrices to make recommendations without having to refer back to the complete data set.
- Latent variables - ingferred, nonobservable variables that are present within, and affect the behavior of a data set.

$\Large \begin{bmatrix} {}_{A} \end{bmatrix} = \begin{bmatrix} {}_{u} \end{bmatrix} \times \begin{bmatrix} {}_{S} \end{bmatrix} \times \begin{bmatrix} {}_{v} \end{bmatrix}$


$\Large {A} = {u} \times {S} \times {v}$

- **A** = Original matrix (utility matrix)
- **u** = Left orthogonal matrix - holds important, non-redundant information about users
- **v** = Right orthogonal matrix - holds important, non-redundant information on items
- **S** = Diagonal matrix - contains all of the information about the decomposition processes performned during the compression

**Building a utility matrix**

```python
ratings_crosstab = combined_movies_data.pivot_table(
    values='rating', 
    index='user_id',
    columns='movie_title',
    fill_value=0)

# shape = (num_users, num_movies)
```

This will generate cross table with users as the rows (indices) and each movie as the columns, a typical wide matrix.

**Transposing the Matrix**

```python
ratings_crosstab.values.T

# shape = (num_movies, num_users)
```

This will transpose the matrix, rows interchange with columns.

**Decomposing the Matrix**

```python
SVD = TruncatedSVD(n_components=12, random_state=17)

resultant_matrix = SVD.fit_transform(X)

# shape = (num_movies, n_components=12)
```

**Generating a Correlation Matrix**

```python
corr_mat = np.corrcoef(resultant_matrix)

# shape = (num_movies, num_movies)
```

**Isolating top movie from the correlation matrix**

```python
movie_names = rating_crosstab.columns
movies_list = list(movie_names)

top_movie = movies_lisst.index('<Top movie title (9999)>')

corr_top_movie = corr_mat[top_movie]
corr_top_movie.shape

# shape = (num_movies,)
```

**Recommending a Highly Correlated Movie**

```python
list(movie_names[
    (corr_top_movie < 1.0) & (corr_top_movie > 0.9)
])

# list of highly correlated movie w.r.t. 'Top movie'
```

## Machine Learning based Recommenders

### Content-based recommender systems

**Content-based recommenders recomend items based on similarities between features.**

*Example: A user who loves Miami might also love Austin, based on the similarities between temperature. const of living and Wi-Fi speeds at both places.*

#### K-nearest neighbor algorithm

- Unsupervised classifier
- Also known as a memory-based system
- Memorizes instances and then recommends item (a single instance) based on how quantitatively similar it is to a new, incoming instance.

Example:
> I want to buy a car that gets 25 MPG, and has a 4.7 L engine with 425 HP.

Solution is to find **"1"** car closest to the specification provided in cartesian distance.


```python
import numpy as np
import pandas as pd

import sklearn
from sklearn.neighbors import NearestNeighbors
```


```python
cars = pd.read_csv('mtcars.csv')
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
cars.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_names</th>
      <th>mpg</th>
      <th>cyl</th>
      <th>disp</th>
      <th>hp</th>
      <th>drat</th>
      <th>wt</th>
      <th>qsec</th>
      <th>vs</th>
      <th>am</th>
      <th>gear</th>
      <th>carb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mazda RX4</td>
      <td>21.0</td>
      <td>6</td>
      <td>160.0</td>
      <td>110</td>
      <td>3.90</td>
      <td>2.620</td>
      <td>16.46</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mazda RX4 Wag</td>
      <td>21.0</td>
      <td>6</td>
      <td>160.0</td>
      <td>110</td>
      <td>3.90</td>
      <td>2.875</td>
      <td>17.02</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Datsun 710</td>
      <td>22.8</td>
      <td>4</td>
      <td>108.0</td>
      <td>93</td>
      <td>3.85</td>
      <td>2.320</td>
      <td>18.61</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hornet 4 Drive</td>
      <td>21.4</td>
      <td>6</td>
      <td>258.0</td>
      <td>110</td>
      <td>3.08</td>
      <td>3.215</td>
      <td>19.44</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hornet Sportabout</td>
      <td>18.7</td>
      <td>8</td>
      <td>360.0</td>
      <td>175</td>
      <td>3.15</td>
      <td>3.440</td>
      <td>17.02</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
specifications_needed = [15, 300, 160, 3.2]
# mpg, disp, hp, wt

X = cars[['mpg', 'disp', 'hp', 'wt']]
X[0:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>disp</th>
      <th>hp</th>
      <th>wt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21.0</td>
      <td>160.0</td>
      <td>110</td>
      <td>2.620</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21.0</td>
      <td>160.0</td>
      <td>110</td>
      <td>2.875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22.8</td>
      <td>108.0</td>
      <td>93</td>
      <td>2.320</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21.4</td>
      <td>258.0</td>
      <td>110</td>
      <td>3.215</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18.7</td>
      <td>360.0</td>
      <td>175</td>
      <td>3.440</td>
    </tr>
  </tbody>
</table>
</div>




```python
nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(X)

y, neighbor_coord = nearest_neighbors.kneighbors([specifications_needed])

print("y = {}, neighbor_coord = {}".format(y, neighbor_coord))
```

    y = [[10.77474942]], neighbor_coord = [[22]]



```python
cars.iloc[neighbor_coord[0]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_names</th>
      <th>mpg</th>
      <th>cyl</th>
      <th>disp</th>
      <th>hp</th>
      <th>drat</th>
      <th>wt</th>
      <th>qsec</th>
      <th>vs</th>
      <th>am</th>
      <th>gear</th>
      <th>carb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>AMC Javelin</td>
      <td>15.2</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3.15</td>
      <td>3.435</td>
      <td>17.3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
cars
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>car_names</th>
      <th>mpg</th>
      <th>cyl</th>
      <th>disp</th>
      <th>hp</th>
      <th>drat</th>
      <th>wt</th>
      <th>qsec</th>
      <th>vs</th>
      <th>am</th>
      <th>gear</th>
      <th>carb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mazda RX4</td>
      <td>21.0</td>
      <td>6</td>
      <td>160.0</td>
      <td>110</td>
      <td>3.90</td>
      <td>2.620</td>
      <td>16.46</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mazda RX4 Wag</td>
      <td>21.0</td>
      <td>6</td>
      <td>160.0</td>
      <td>110</td>
      <td>3.90</td>
      <td>2.875</td>
      <td>17.02</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Datsun 710</td>
      <td>22.8</td>
      <td>4</td>
      <td>108.0</td>
      <td>93</td>
      <td>3.85</td>
      <td>2.320</td>
      <td>18.61</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hornet 4 Drive</td>
      <td>21.4</td>
      <td>6</td>
      <td>258.0</td>
      <td>110</td>
      <td>3.08</td>
      <td>3.215</td>
      <td>19.44</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hornet Sportabout</td>
      <td>18.7</td>
      <td>8</td>
      <td>360.0</td>
      <td>175</td>
      <td>3.15</td>
      <td>3.440</td>
      <td>17.02</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Valiant</td>
      <td>18.1</td>
      <td>6</td>
      <td>225.0</td>
      <td>105</td>
      <td>2.76</td>
      <td>3.460</td>
      <td>20.22</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Duster 360</td>
      <td>14.3</td>
      <td>8</td>
      <td>360.0</td>
      <td>245</td>
      <td>3.21</td>
      <td>3.570</td>
      <td>15.84</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Merc 240D</td>
      <td>24.4</td>
      <td>4</td>
      <td>146.7</td>
      <td>62</td>
      <td>3.69</td>
      <td>3.190</td>
      <td>20.00</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Merc 230</td>
      <td>22.8</td>
      <td>4</td>
      <td>140.8</td>
      <td>95</td>
      <td>3.92</td>
      <td>3.150</td>
      <td>22.90</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Merc 280</td>
      <td>19.2</td>
      <td>6</td>
      <td>167.6</td>
      <td>123</td>
      <td>3.92</td>
      <td>3.440</td>
      <td>18.30</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Merc 280C</td>
      <td>17.8</td>
      <td>6</td>
      <td>167.6</td>
      <td>123</td>
      <td>3.92</td>
      <td>3.440</td>
      <td>18.90</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Merc 450SE</td>
      <td>16.4</td>
      <td>8</td>
      <td>275.8</td>
      <td>180</td>
      <td>3.07</td>
      <td>4.070</td>
      <td>17.40</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Merc 450SL</td>
      <td>17.3</td>
      <td>8</td>
      <td>275.8</td>
      <td>180</td>
      <td>3.07</td>
      <td>3.730</td>
      <td>17.60</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Merc 450SLC</td>
      <td>15.2</td>
      <td>8</td>
      <td>275.8</td>
      <td>180</td>
      <td>3.07</td>
      <td>3.780</td>
      <td>18.00</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Cadillac Fleetwood</td>
      <td>10.4</td>
      <td>8</td>
      <td>472.0</td>
      <td>205</td>
      <td>2.93</td>
      <td>5.250</td>
      <td>17.98</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Lincoln Continental</td>
      <td>10.4</td>
      <td>8</td>
      <td>460.0</td>
      <td>215</td>
      <td>3.00</td>
      <td>5.424</td>
      <td>17.82</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Chrysler Imperial</td>
      <td>14.7</td>
      <td>8</td>
      <td>440.0</td>
      <td>230</td>
      <td>3.23</td>
      <td>5.345</td>
      <td>17.42</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Fiat 128</td>
      <td>32.4</td>
      <td>4</td>
      <td>78.7</td>
      <td>66</td>
      <td>4.08</td>
      <td>2.200</td>
      <td>19.47</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Honda Civic</td>
      <td>30.4</td>
      <td>4</td>
      <td>75.7</td>
      <td>52</td>
      <td>4.93</td>
      <td>1.615</td>
      <td>18.52</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Toyota Corolla</td>
      <td>33.9</td>
      <td>4</td>
      <td>71.1</td>
      <td>65</td>
      <td>4.22</td>
      <td>1.835</td>
      <td>19.90</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Toyota Corona</td>
      <td>21.5</td>
      <td>4</td>
      <td>120.1</td>
      <td>97</td>
      <td>3.70</td>
      <td>2.465</td>
      <td>20.01</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Dodge Challenger</td>
      <td>15.5</td>
      <td>8</td>
      <td>318.0</td>
      <td>150</td>
      <td>2.76</td>
      <td>3.520</td>
      <td>16.87</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>AMC Javelin</td>
      <td>15.2</td>
      <td>8</td>
      <td>304.0</td>
      <td>150</td>
      <td>3.15</td>
      <td>3.435</td>
      <td>17.30</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Camaro Z28</td>
      <td>13.3</td>
      <td>8</td>
      <td>350.0</td>
      <td>245</td>
      <td>3.73</td>
      <td>3.840</td>
      <td>15.41</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Pontiac Firebird</td>
      <td>19.2</td>
      <td>8</td>
      <td>400.0</td>
      <td>175</td>
      <td>3.08</td>
      <td>3.845</td>
      <td>17.05</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Fiat X1-9</td>
      <td>27.3</td>
      <td>4</td>
      <td>79.0</td>
      <td>66</td>
      <td>4.08</td>
      <td>1.935</td>
      <td>18.90</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Porsche 914-2</td>
      <td>26.0</td>
      <td>4</td>
      <td>120.3</td>
      <td>91</td>
      <td>4.43</td>
      <td>2.140</td>
      <td>16.70</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Lotus Europa</td>
      <td>30.4</td>
      <td>4</td>
      <td>95.1</td>
      <td>113</td>
      <td>3.77</td>
      <td>1.513</td>
      <td>16.90</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Ford Pantera L</td>
      <td>15.8</td>
      <td>8</td>
      <td>351.0</td>
      <td>264</td>
      <td>4.22</td>
      <td>3.170</td>
      <td>14.50</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Ferrari Dino</td>
      <td>19.7</td>
      <td>6</td>
      <td>145.0</td>
      <td>175</td>
      <td>3.62</td>
      <td>2.770</td>
      <td>15.50</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Maserati Bora</td>
      <td>15.0</td>
      <td>8</td>
      <td>301.0</td>
      <td>335</td>
      <td>3.54</td>
      <td>3.570</td>
      <td>14.60</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Volvo 142E</td>
      <td>21.4</td>
      <td>4</td>
      <td>121.0</td>
      <td>109</td>
      <td>4.11</td>
      <td>2.780</td>
      <td>18.60</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
