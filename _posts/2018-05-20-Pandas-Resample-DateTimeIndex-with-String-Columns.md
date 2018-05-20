---
layout: post
title: "How to Resample with Strings to Label Data"
date: 2018-05-20
---
# How to Resample with Strings to Label Data


```python
data = pd.read_csv('data.csv', index_col=0)
data.index = pd.to_datetime(data.index)
```


```python
import datetime
session = datetime.datetime.strptime('2018-05-14', '%Y-%m-%d').date()
data = data.loc[data.index.date==session]
data.head()
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
      <th>Label</th>
      <th>User ID</th>
      <th>Engagement</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-05-14 10:40:15</th>
      <td>None</td>
      <td>None</td>
      <td>-2.343828</td>
    </tr>
    <tr>
      <th>2018-05-14 10:40:16</th>
      <td>None</td>
      <td>None</td>
      <td>-14.431959</td>
    </tr>
    <tr>
      <th>2018-05-14 10:40:16</th>
      <td>None</td>
      <td>None</td>
      <td>39.127761</td>
    </tr>
    <tr>
      <th>2018-05-14 10:40:17</th>
      <td>None</td>
      <td>None</td>
      <td>16.067326</td>
    </tr>
    <tr>
      <th>2018-05-14 10:40:18</th>
      <td>None</td>
      <td>None</td>
      <td>9.930275</td>
    </tr>
  </tbody>
</table>
</div>



We have a DataFrame with a DateTimeIndex, two columns with string values (Label and User ID) and a numerical column (Engagement). Each observation occurs about once per second, although sometimes there are multiple observations per second. Pandas has a resample method that allows for easily changing the time scale of the index. For example, we can resample by '1min' to get the Engagement scores per minute.


```python
%matplotlib inline
data.resample('1min').mean().plot(kind='line', rot=45, marker='o')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x28dc8e34d30>




![png](/images/Pandas-Resample-DateTimeIndex-with-String-Columns_files/Pandas-Resample-DateTimeIndex-with-String-Columns_4_1.png)


By default, Pandas does not know to handle strings in the resample function. What if we want to use those strings as data labels? Here's how. We compute the mode of the string per time period.

# Compute the mode of the User ID and Label


```python
from scipy.stats import mode as md
col = 'Engagement'
x = data.resample('1min').agg({col:'mean',
                                    "User ID":lambda x: md(x)[0],#
                                    "Label":lambda x: md(x)[0]})
x['User ID'] = [x if isinstance(x, str) else np.nan for x in x['User ID']]
x['Label'] = [x if isinstance(x, str) else np.nan for x in x['Label']]
x.head()
```

    c:\users\john maxi\anaconda3\envs\tensorflowenv\lib\site-packages\scipy\stats\stats.py:245: RuntimeWarning: The input array could not be properly checked for nan values. nan values will be ignored.
      "values. nan values will be ignored.", RuntimeWarning)
    




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
      <th>Label</th>
      <th>Engagement</th>
      <th>User ID</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-05-14 10:40:00</th>
      <td>None</td>
      <td>14.068135</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2018-05-14 10:41:00</th>
      <td>None</td>
      <td>0.231592</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2018-05-14 10:42:00</th>
      <td>demo</td>
      <td>-2.981444</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2018-05-14 10:43:00</th>
      <td>demo</td>
      <td>12.820799</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2018-05-14 10:44:00</th>
      <td>demo</td>
      <td>8.654669</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



To plot the data using our new group labels we can pivot around one of the label columns.

# Pivot to make User ID the columns


```python
a = x.pivot(columns='User ID', values=col)
```

# Pivot to make Label the columns


```python
y = x.pivot(columns='Label', values=col)
```

# Plot the data labeled by User ID


```python
a.plot(kind='line');
```


![png](/images/Pandas-Resample-DateTimeIndex-with-String-Columns_files/Pandas-Resample-DateTimeIndex-with-String-Columns_14_0.png)


# Plot the data labeled by Label


```python
y.plot(kind='line');
```


![png](/images/Pandas-Resample-DateTimeIndex-with-String-Columns_files/Pandas-Resample-DateTimeIndex-with-String-Columns_16_0.png)

