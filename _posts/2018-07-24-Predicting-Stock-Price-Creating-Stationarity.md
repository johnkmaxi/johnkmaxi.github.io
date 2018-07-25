---
layout: post
title: "Predicting Stock Price Creating Stationarity"
date: 2018-07-24
---
# How to build time series models - Accounting for Non-Stationary Data
In a previous [post](https://johnkmaxi.github.io/blog/2018/05/31/Predicting-Stock-Price), I demonstrated how to load time series data, normalize the data, and build a model to predict future values of the time series. That post was more about the formatting and model building. Today, I'd like to extend that work by diving deeper in order to improve model performance.


# Import data


```python
import quandlkey as qk
data = qk.quandl_stocks('AAPL') # data from 2000/1/1 to today
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
      <th>WIKI/AAPL - Open</th>
      <th>WIKI/AAPL - High</th>
      <th>WIKI/AAPL - Low</th>
      <th>WIKI/AAPL - Close</th>
      <th>WIKI/AAPL - Volume</th>
      <th>WIKI/AAPL - Ex-Dividend</th>
      <th>WIKI/AAPL - Split Ratio</th>
      <th>WIKI/AAPL - Adj. Open</th>
      <th>WIKI/AAPL - Adj. High</th>
      <th>WIKI/AAPL - Adj. Low</th>
      <th>WIKI/AAPL - Adj. Close</th>
      <th>WIKI/AAPL - Adj. Volume</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2000-01-03</th>
      <td>104.87</td>
      <td>112.50</td>
      <td>101.69</td>
      <td>111.94</td>
      <td>4783900.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.369314</td>
      <td>3.614454</td>
      <td>3.267146</td>
      <td>3.596463</td>
      <td>133949200.0</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>108.25</td>
      <td>110.62</td>
      <td>101.19</td>
      <td>102.50</td>
      <td>4574800.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.477908</td>
      <td>3.554053</td>
      <td>3.251081</td>
      <td>3.293170</td>
      <td>128094400.0</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>103.75</td>
      <td>110.56</td>
      <td>103.00</td>
      <td>104.00</td>
      <td>6949300.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.333330</td>
      <td>3.552125</td>
      <td>3.309234</td>
      <td>3.341362</td>
      <td>194580400.0</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>106.12</td>
      <td>107.00</td>
      <td>95.00</td>
      <td>95.00</td>
      <td>6856900.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.409475</td>
      <td>3.437748</td>
      <td>3.052206</td>
      <td>3.052206</td>
      <td>191993200.0</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>96.50</td>
      <td>101.00</td>
      <td>95.50</td>
      <td>99.50</td>
      <td>4113700.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.100399</td>
      <td>3.244977</td>
      <td>3.068270</td>
      <td>3.196784</td>
      <td>115183600.0</td>
    </tr>
  </tbody>
</table>
</div>



There are several columns of data available to work with and using some combination of them is likely to produce the best results. For this demonstration, we're going to use just one column, the Adjusted Close.


```python
# Only need to work with the adj. close
close = data['WIKI/AAPL - Adj. Close']
```

The problem we are trying to solve in this demonstration is to predict the stock price on a given day by using the price from the previous two days. To make this problem more conducive for use with a machine learning model we need to do some data normalizing. When working with financial data, it is standard practice to convert the actual price into daily percent returns (DR). We do this by dividing the stock price on day $x_t$ by the price from the day before, $x_{t-1}$:

$$ DR = \frac{x_t}{x_{t-1}}$$

# Convert Adj. Close to % Change from Previous Day


```python
# convert the price to a % change from the previous day
close = (close/close.shift(1))
```

Linear regression models typically assume that the underlying data is stationary, i.e., that the underlying distribution of the date is stable. However, with time series data, it is often the case that the mean and standard deviation change over time. We intuitively know that the price of stocks changes over time, but does the average percent change in price change ove time as well? Let's use a 20 day rolling mean as our metric. A stationary time series will have a mean that is constant over time, whereas non-stationary data will have a fluctuating mean.


```python
fig, ax = plt.subplots(1, figsize=(9,5))
close.rolling(20).mean().plot(kind='line', ax=ax)
ax.legend(['20 Day Average % Price Change'])
ax.set_ylabel('% Change from Previous Day')
for tick in ax.get_xticklabels():
    tick.set_fontsize(12)
for tick in ax.get_yticklabels():
    tick.set_fontsize(12)
plt.show()
```


![png](/images/Predicting-Stock-Price-Creating-Stationarity_files/Predicting-Stock-Price-Creating-Stationarity_8_0.png)


We can clearly see some peaks and valleys in the roling mean. There are also downward and upward trends present throughout the plot. There is a clear downward trend leading up to 2003, followed by and upward trend leading up to 2005. The non-stationarity of the data will lead to poor performance by linear modeling techniques (as we saw previously). One of the ways to address this issue to look for transformations of the data produce stationary representations of the data. This kind of technique is what the difference factor does in ARIMA models. Here, I would like to explore a transformation I came upon serendipitously. I call it the reverse second order percent change. After calculating the percent change in closing price above, we use that transform again, but in reverse order. That is, we divide yesterday's percent change by today's percent change and subtract 1 to remove the bias.


```python
close = (close.shift(1)/close)-1
```


```python
fig, ax = plt.subplots(1, figsize=(9,5))
close.rolling(20).mean().plot(kind='line', ax=ax)
ax.legend(['20 Day Average Reverse 2nd Order % Price Change'])
ax.set_ylabel('% Change from Previous Day')
for tick in ax.get_xticklabels():
    tick.set_fontsize(12)
for tick in ax.get_yticklabels():
    tick.set_fontsize(12)
plt.show()
```


![png](/images/Predicting-Stock-Price-Creating-Stationarity_files/Predicting-Stock-Price-Creating-Stationarity_11_0.png)


The rolling average of the mean still has a few peaks and valleys, but the trends we saw in the above plot have mostly been removed. We can now follow the same processing steps from the [previous post](https://johnkmaxi.github.io/blog/2018/05/31/Predicting-Stock-Price) to see how this extra transformation affects our predictive ability.

# Create a time lag DataFrame

In this example, we will use the previous 2 days daily return to predict the predict the daily return for the next day. The primary reason for making the model this way is to allow for easy visualization.

Pandas has a built in `shift()` method for just this purpose. `shift` takes an integer as an argument indicating how many positions to shift the data by. Negative values shift the data in the reverse direction. See the [docs](http://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.shift.html) for other arguments `shift` can take.


```python
# use close to shift(1)
close = pd.concat([close,close.shift(1),close.shift(2)],axis=1)
# rename the columns
close.columns = ['Y','X1','X2']
# get rid of NaN values after using shift
close.dropna(inplace=True)
close.head()
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
      <th>Y</th>
      <th>X1</th>
      <th>X2</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-07</th>
      <td>-0.127851</td>
      <td>0.110757</td>
      <td>-0.097538</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>0.066119</td>
      <td>-0.127851</td>
      <td>0.110757</td>
    </tr>
    <tr>
      <th>2000-01-11</th>
      <td>0.035372</td>
      <td>0.066119</td>
      <td>-0.127851</td>
    </tr>
    <tr>
      <th>2000-01-12</th>
      <td>0.009356</td>
      <td>0.035372</td>
      <td>0.066119</td>
    </tr>
    <tr>
      <th>2000-01-13</th>
      <td>-0.152834</td>
      <td>0.009356</td>
      <td>0.035372</td>
    </tr>
  </tbody>
</table>
</div>



# Visualize the data

Let's take a look at the values we are trying to predict. Visualizing the data is an important step in the modeling processing, esspecially during exploratory data analysis. Visualizing the data helps in developing an intuition of the data. To start, we'll plot the target values as a line plot and the two features as a scatter plot. We'll color each dot in the feature based on whether the daily return is positive (green) or negative (red) to see if there is any visual separation of the features with respect to the direction of the daily return.


```python
%matplotlib inline
fig, ax = plt.subplots(2, figsize=(16,9))
close['Y'].plot(kind='line', ax=ax[0])
ax[1].scatter(close['X1'],close['X2'],c=['r' if x<0 else 'g' for x in close['Y']])
ax[1].set_xlim(-0.25,0.25) # set the plot limits to zoom in on the bulk of the data
ax[1].set_ylim(-0.25,0.25)
ax[0].set_title('Reverse 2nd Order % Change from Previous Day', fontsize=20)
ax[0].set_ylabel('% Change', fontsize=18)
ax[1].set_title('One Day Ago vs Two Days Ago', fontsize=20)
ax[1].set_xlabel('X1', fontsize=18)
ax[1].set_ylabel('X2', fontsize=18)
for a in ax:
    for tick in a.get_xticklabels():
        tick.set_fontsize(12)
    for tick in a.get_yticklabels():
        tick.set_fontsize(12)
fig.tight_layout();
```


![png](/images/Predicting-Stock-Price-Creating-Stationarity_files/Predicting-Stock-Price-Creating-Stationarity_15_0.png)


The top plot shows the same thing as our rolling mean data above. The bottom plot shows that our two features seem to do a good job of separating the data. The green dots represent the features associated with positive y values, and the red dots show the features associated with negative y values. We can visually see that there is a diagonal line that separates these points, giving us some evidence that a linear model can spearate and make good predictiosn using these features. We'll now train the same model as before. Again, the only difference is that the data has been transformed to create stationarity.

# Split into train and test set


```python
# we'll use the first 60% of the data as a training set
split_index = int(len(close)*0.6)
train = close.iloc[:split_index,:]
test = close.iloc[split_index:,:]
ytrain = train['Y']
Xtrain = train.drop('Y', axis=1)
ytest = test['Y']
Xtest = test.drop('Y', axis=1)
```

# Train the model

For simplicity, we'll use a linear regression model. After training, we'll have a set of weights and a bias that we can use to calculate new daily returns with the formula

$$ R2DR = w_1x_1 + w_2x_2 + b$$

where $w_1$ and $w_2$ are the coefficients by which we multiply the $R2DR$ from one and two days ago. $b$ is the bias term which sets the x-intercept for the equation.

Sklearn is usually my go to for model development, keep in mind that numpy and scipy both have methods for fitting linear regressions. When speed counts, those libraries tend to be a little faster. The code below imports the LinearRegression class, instantiates it, fits the model to the data, and then prints out $w_1$, $w_2$, and $b$.


```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Xtrain, ytrain)
print(lr.coef_, lr.intercept_)
```

    [-0.61253731 -0.28064298] 0.00208474021198
    

# Score the model

Now we need to find out if our model is any good. To do that we score the model against the test data. The mean squared error is a commonly used metric for regression problems. It is computed by taking the average over all points of the squared difference between each corresponding target value and prediction:

$$MSE = \frac{1}{n}\Sigma(\hat{y} - y)^2$$

where $\hat{y}$ is the predicted value and $y$ is the know target.

Sklearn uses the $R^2$ value as the built in way to evaluate regression models. It typically ranges from 0 to 1, although negative scores are possible. The $R^2$ value represents how well the model matches the predictions compared to a model that only predicts the mean of all the values. Therefore, a score of 1 indicates perfect predictions, a score of 0 indicates the model is no better than simply predicting the mean of all the target points, and negative values indicate the model performs worse than a mean model. One caveat of the $R^2$ is that it will increase as more predictors are added to the model. If using $R^2$ as the sole scoring metric be sure to take this into account by using the adjusted $R^2$ in conjuction with Mallow's Cp to account for the number of factors in the model.


```python
from sklearn.metrics import mean_squared_error, r2_score
# the R^2 value, 1.0 = perfect predictions, 0 = constant value, < 0 = worse than constant prediction
print('R^2: ', lr.score(Xtest, ytest))
# MSE
print('MSE: ', mean_squared_error(ytest,lr.predict(Xtest)))
```

    R^2:  0.302889946379
    MSE:  0.000338446670654
    


```python
# Baseline regressors
base_predictor = [np.mean(ytrain) for x in ytest]
print('R^2: ', r2_score(ytest, base_predictor))
print('MSE: ', mean_squared_error(ytest, base_predictor))
```

    R^2:  -0.00143220646302
    MSE:  0.000486194962191
    

The $R^2$ value indicates that the model accounts for about 30% of variabilty in the target values. This values is much better than the $R^2$ value of the baseline classifier and also much better than the $R^2$ score (-0.004) when we did not address the non-stationarity of the data. 

Let's look at how the model predictions compare to the target values.


```python
fig, ax = plt.subplots(1, figsize=(16,9))
ytest.plot(kind='line',ax=ax, label='Targets')
ax.plot(Xtest.index, lr.predict(Xtest.values), label='Predictions');
ax.legend()
for tick in a.get_xticklabels():
    tick.set_fontsize(15)
for tick in a.get_yticklabels():
    tick.set_fontsize(15)
```


![png](/images/Predicting-Stock-Price-Creating-Stationarity_files/Predicting-Stock-Price-Creating-Stationarity_25_0.png)


In this post, you learned how to identify and deal with non-stationary time series data. Transforming time series data to become stationary can improve the performance of your models.

Thanks for reading!
