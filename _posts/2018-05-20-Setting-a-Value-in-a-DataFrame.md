---
layout: post
title: "Setting a Value in a DataFrame"
date: 2018-05-20
---
# Create DataFrame 


```python
index_x = ['a', 'b', 'c', 'd']

columns = ['one', 'two', 'three', 'four']

M = pd.DataFrame(np.random.randn(4,4), index=index_x, columns=columns)
M
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.227063</td>
      <td>-1.238476</td>
      <td>0.260084</td>
      <td>-0.693411</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.166195</td>
      <td>-0.250198</td>
      <td>-0.376468</td>
      <td>0.116053</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.147651</td>
      <td>-1.398541</td>
      <td>-0.430007</td>
      <td>1.121594</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1.346145</td>
      <td>0.743654</td>
      <td>1.905355</td>
      <td>1.237901</td>
    </tr>
  </tbody>
</table>
</div>



# Set a value using a row/column indicator


```python
M.set_value('d','two',5.0)
M
```

    c:\users\john maxi\anaconda3\envs\tensorflowenv\lib\site-packages\ipykernel_launcher.py:1: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
      """Entry point for launching an IPython kernel.
    




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
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.227063</td>
      <td>-1.238476</td>
      <td>0.260084</td>
      <td>-0.693411</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.166195</td>
      <td>-0.250198</td>
      <td>-0.376468</td>
      <td>0.116053</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.147651</td>
      <td>-1.398541</td>
      <td>-0.430007</td>
      <td>1.121594</td>
    </tr>
    <tr>
      <th>d</th>
      <td>1.346145</td>
      <td>5.000000</td>
      <td>1.905355</td>
      <td>1.237901</td>
    </tr>
  </tbody>
</table>
</div>


