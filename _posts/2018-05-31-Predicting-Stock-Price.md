---
layout: post
title: "Preparing Time Series Data For Machine Learning - Predicting Stock Price"
date: 2018-05-31
---
# How to build time series models
In this post, I will demonstrate how to load time series data, normalize the data, and build a model to predict future values of the time series. For this example, I'm going to use historical Apple stock data. You can get the data yourself by getting a Quandl API key. See this [post](https://johnkmaxi.github.io/blog/2018/04/22/Bollinger-Band-Trading-with-Apple-Stock) for a the nuts and bolts of the quandl_stocks function. The first thing we'll do is grab the data.

The purpose of this post is to show a quick example of getting and formatting the data. We're going to use a really simple linear regression model, which as you will see, actually does a really terrible job on this particular problem. However, we can use the foundation we build here with other types of models or feature engineering to improve the performance of the model.

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
close = (close/close.shift(1))-1
```

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
      <th>2000-01-06</th>
      <td>-0.086538</td>
      <td>0.014634</td>
      <td>-0.084331</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>0.047368</td>
      <td>-0.086538</td>
      <td>0.014634</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>-0.017588</td>
      <td>0.047368</td>
      <td>-0.086538</td>
    </tr>
    <tr>
      <th>2000-01-11</th>
      <td>-0.051151</td>
      <td>-0.017588</td>
      <td>0.047368</td>
    </tr>
    <tr>
      <th>2000-01-12</th>
      <td>-0.059946</td>
      <td>-0.051151</td>
      <td>-0.017588</td>
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
ax[1].set_xlim(-0.2,0.2) # set the plot limits to zoom in on the bulk of the data
ax[1].set_ylim(-0.2,0.2)
ax[0].set_title('% Change from Previous Day', fontsize=20)
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


![png](/images/Predicting-Stock-Price_files/Predicting-Stock-Price_10_0.png)


Our visualizations show that there is a lot of fluctuation in the day to day return of the stock. On this scale no apparaent pattern or trends are clearly evident. We should also notice that there is no separation of days when the price went up or down using our two input features. This is a big red flag indicating that that we aren't likely to have good model performance, especially with a linear-type model we are using here. We would want to pick a model that has more parameters or can learn non-linear features (such as an artifical neural network or a decision tree/random forest).

For now, let's move on to using our data to train a model. It's important in time series modeling to not include future events in the training data. if we do, we are essentially looking into the future during training, resulting in inflated model performance. For this reason, the train/test split is usually done by including all values up to a certain point in the training set, and all remaining values in the test set.

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

$$ DR = w_1x_1 + w_2x_2 + b$$

where $w_1$ and $w_2$ are the coefficients by which we multiply the $DR$ from one and two days ago. $b$ is the bias term which sets the x-intercept for the equation.

Sklearn is usually my go to for model development, keep in mind that numpy and scipy both have methods for fitting linear regressions. When speed counts, those libraries tend to be a little faster. The code below imports the LinearRegression class, instantiates it, fits the model to the data, and then prints out $w_1$, $w_2$, and $b$.


```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Xtrain, ytrain)
print(lr.coef_, lr.intercept_)
```

    [-0.04024398 -0.016456  ] 0.00150257519941
    

# Score the model

Now we need to find out if our model is any good. To do that we score the model against the test data. The mean squared error is a commonly used metric for regression problems. It is computed by taking the average over all points of the squared difference between each corresponding target value and prediction:

$$MSE = \frac{1}{n}\Sigma(\hat{y} - y)$$

where $\hat{y}$ is the predicted value and $y$ is the know target.

Sklearn uses the $R^2$ value as the built in way to evaluate regression models. It typically ranges from 0 to 1, although negative scores are possible. The $R^2$ value represents how well the model matches the predictions compared to a model that only predicts the mean of all the values. Therefore, a score of 1 indicates perfect predictions, a score of 0 indicates the model is no better than simply predicting the mean of all the target points, and negative values indicate the model performs worse than a mean model. One caveat of the $R^2$ is that it will increase as more predictors are added to the model. If using $R^2$ as the sole scoring metric be sure to take this into account by using the adjusted $R^2$ in conjuction with Mallow's Cp to account for the number of factors in the model.


```python
from sklearn.metrics import mean_squared_error, r2_score
# the R^2 value, 1.0 = perfect predictions, 0 = constant value, < 0 = worse than constant prediction
print('R^2: ', lr.score(Xtest, ytest))
# MSE
print('MSE: ', mean_squared_error(ytest,lr.predict(Xtest)))
```

    R^2:  -0.00450841891829
    MSE:  0.000249952056255
    


```python
# Baseline regressors
base_predictor = [np.mean(ytrain) for x in ytest]
print('R^2: ', r2_score(ytest, base_predictor))
print('MSE: ', mean_squared_error(ytest, base_predictor))
```

    R^2:  -0.00112004188527
    MSE:  0.000249108925634
    

The R^2 value indicates that the model actually does worse than simply predicting the mean of y for every prediction. A negative R^2 value can also arise when when the model is not fitted on the data being tested, as is the case here (because of train/test split). The MSE looks really small, but it is actually larger than the baseline classifier's MSE. It is important to have a baseline classifier to put the model's results in context. Another good strategy is to use the root mean squared error. This method takes the square root of the MSE which puts the error on the same scale as the original values. In this case, the RMSE is


```python
print('RMSE: ', mean_squared_error(ytest, base_predictor)**0.5)
print('Average change in daily return: ', ytest.abs().mean())
```

    RMSE:  0.0157831849015
    Average change in daily return:  0.011256048038
    

The RMSE tells us that we our predictions were off by about 1.5% on average. This is a pretty huge amount of error considering that the average change in daily returns was only 1.1%. Our error is larger in magnitude than the values we are trying to predict. A linear model with these two features is clearly not a good choice for modeling the data used here. Let's visualize the predictions compared targets to keep developing our intuition of what went wrong.

# Plot the Predictions


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


![png](/images/Predicting-Stock-Price_files/Predicting-Stock-Price_22_0.png)


What stands out from this plot, is that the predicted values are on a much smaller scale than the target values. This gives us a nice visual explanation of why our model did worse than the baseline model of just predicting the mean of the training data targets. 

We saw earlier that X1 and X2 do linearly separate Y into postive or negative values. For our final diagnosis of this model, let's look at how X1 and X2 individually correlate with Y and compare the distributions of our target and our predictions in the test data.

# Features vs Target, Target Distribution


```python
from matplotlib import gridspec
fig = plt.figure(figsize=(16,9))
spec = gridspec.GridSpec(ncols=2, nrows=2)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[1, :])
ax1.scatter(close['X1'],close['Y'])
ax1.set_title('X1 vs Y', fontsize=20)
ax2.scatter(close['X2'],close['Y'])
ax2.set_title('X2 vs Y', fontsize=20)
ax3.hist(lr.predict(close.loc[:,['X1','X2']]),50, alpha=0.5, label='Predictions');
ax3.hist(close['Y'],50, alpha=0.5, label='Targets');
ax3.set_title('Actual and Predicted Distributions', fontsize=20);
ax3.legend();
for a in [ax1,ax2,ax3]:
    for tick in a.get_xticklabels():
        tick.set_fontsize(15)
    for tick in a.get_yticklabels():
        tick.set_fontsize(15)
fig.tight_layout();
```


![png](/images/Predicting-Stock-Price_files/Predicting-Stock-Price_24_0.png)


First, note that there isn't any relationship between either of our predictor variables X1 or X2 with Y. That isn't necessarily a bad thing, as features that are independent of each other tend to give better model performance. However, this linear model isn't able to derive any useful information from these features. Second, the histogram in the bottom panel really shows how small the variability of our model predictions are. they essentially only predicted the mean (although slightly worse, as mention above). 

So our model didn't work! That's not a bad thing. Model building  and machine learning involves lots of experimentation to get a good outcome; this is called data **science** after all. So knowing the outcome of this experiment, what should our next steps be? Well, we learned that our model isn't able to acount for enough of the variability in the target values. Therfore, our next step should be to increase model complexity. This could be accomplished by adding more features, adding polynomial features. We could also try deep learning techniques such as Long Short-Term Memory neural networks, which are a good choice for time series regression problems.

I hoped you learned something!
