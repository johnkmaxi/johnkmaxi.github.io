---
layout: post
title: "Time Series Forecast Models ARIMA"
date: 2018-07-25
---
# Time Series Forecasting Models: ARIMA

In this post, you will learn the steps to create ARIMA models for time series forecasting. Using these models, you will be able to make predictions for the future values of things like hotel bookings, product sales, and any other times series, using only historical data of that value. Let's get started!

## ARIMA Models

ARIMA stands for autoregressive, difference, moving average. ARIMA models use at least one of these pieces of informatino to model trends in a time series data. There are four major steps to creating a good ETS model:
1. Create time series decomposition plot
2. Determine the ARIMA terms
3. Build and validate the ARIMA model
4. Forecast

Let's walk through the details of each step.

### Create time series decomposition plot

See my previous post on [ETS models](https://johnkmaxi.github.io/blog/2018/07/25/Time-Series-Forecasting-Models-ETS) for an explanation of the time series decomposition plot. For now, we'll just skip straight to the code to create the plots


```python
# load modules
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa import holtwinters
from statsmodels.tsa import stattools, statespace
from matplotlib import gridspec
```


```python
# load and setup data
data = pd.read_csv('champagne-sales.csv')
# there are 96 observations representing 8 years of monthly data
# we are chooseing an arbitrary start date
data.index = pd.date_range(start='2010-01-01', periods=len(data), freq='m')
# drop the month column since the we have replaced it with the index
data.drop('Month',axis=1,inplace=True)
```


```python
# split the data into train and test to evaluate performance
train, test = data.iloc[:90,0], data.iloc[90:,0]
```


```python
# create additive plot and multiplicative plot side by side
additive = sm.tsa.seasonal_decompose(data['Champagne Sales'], model='add', freq=12)
multiplicative = sm.tsa.seasonal_decompose(data['Champagne Sales'], model='mult', freq=12)
```


```python
fig = plt.figure(figsize=(16,9))
spec = gridspec.GridSpec(ncols=2, nrows=4)
ax1 = fig.add_subplot(spec[0, :])
ax1.set_ylabel('Sales ($)')
ax1.set_title('Champagne Sales')
data.plot(kind='line', ax=ax1)
ax2 = fig.add_subplot(spec[1,0])
ax2.plot(additive.trend)
ax2.set_ylabel('Trend')
ax2.set_title('Additive')
ax3 = fig.add_subplot(spec[2,0])
ax3.plot(additive.seasonal)
ax3.set_ylabel('Season')
ax4 = fig.add_subplot(spec[3,0])
ax4.plot(additive.resid)
ax4.set_ylabel('Residual')
ax5 = fig.add_subplot(spec[1,1])
ax5.plot(multiplicative.trend)
ax5.set_title('Multiplicative')
ax6 = fig.add_subplot(spec[2,1])
ax6.plot(multiplicative.seasonal)
ax7 = fig.add_subplot(spec[3,1])
ax7.plot(multiplicative.resid)
plt.tight_layout()
plt.show();
```


![png](/images/Time-Series-Forecasting-Models-ARIMA_files/Time-Series-Forecasting-Models-ARIMA_5_0.png)


### Determine the ARIMA terms

ARIMA models can be seasonal or non-seasonal. The seaonal plot subplots above indicate that using a seasonal model would be a good choice for this data. The terms we need to determine are for the seaonal component are P, D, and Q. The non-seasonal components are p, d, and q. The p's indicate the number of periods to for, d is for the number of differences needed to make the time series stationary, and q is the lag of the moving average component. Our model will be noted (p,d,q)(P,D,Q)[S], where S is the length of the seasonal period.

The first step in filling out our model terms is to determine the number of differences it requires to make the data staionary. To illustrate this, we'll make 4 plots: the original data, the first difference, the second difference, and the third difference. On each plot we'll include a rolling mean rolling variance line to visualize the stationarity.


```python
fig, ax = plt.subplots(4,3, figsize=(9,9), sharex=True)
plotdata = data.copy()
plotdata['1st Difference'] = plotdata['Champagne Sales'] - plotdata['Champagne Sales'].shift(1)
plotdata['2nd Difference'] = plotdata['1st Difference'] - plotdata['1st Difference'].shift(1)
plotdata['3rd Difference'] = plotdata['2nd Difference'] - plotdata['2nd Difference'].shift(1)
plotdata['Champagne Sales'].plot(kind='line', ax=ax[0][0])
plotdata['Champagne Sales'].rolling(window=12).mean().plot(ax=ax[0][1])
plotdata['Champagne Sales'].rolling(window=12).std().plot(ax=ax[0][2])
plotdata['1st Difference'].plot(kind='line', ax=ax[1][0])
plotdata['1st Difference'].rolling(window=12).mean().plot(ax=ax[1][1])
plotdata['1st Difference'].rolling(window=12).std().plot(ax=ax[1][2])
plotdata['2nd Difference'].plot(kind='line', ax=ax[2][0])
plotdata['2nd Difference'].rolling(window=12).mean().plot(ax=ax[2][1])
plotdata['2nd Difference'].rolling(window=12).std().plot(ax=ax[2][2])
plotdata['3rd Difference'].plot(kind='line', ax=ax[3][0])
plotdata['3rd Difference'].rolling(window=12).mean().plot(ax=ax[3][1])
plotdata['3rd Difference'].rolling(window=12).std().plot(ax=ax[3][2])
ax[0][0].set_title('Time Series')
ax[0][1].set_title('Rolling Mean')
ax[0][2].set_title('Rolling Std Dev')
ax[0][0].set_ylabel('Original')
ax[1][0].set_ylabel('1st')
ax[2][0].set_ylabel('2nd')
ax[3][0].set_ylabel('3rd')
plt.show()
```


![png](/images/Time-Series-Forecasting-Models-ARIMA_files/Time-Series-Forecasting-Models-ARIMA_7_0.png)


Looking at the plots, we can see that the time series looks like it starts to become stationary after 2 differences, based on the rolling mean plot. However, there is a clear trend, indicating non-stationarity in the rolling standard deviation plots for all the differences. What's the deal? The problem is that we have not taken the seasonality of this dataset into account. We need to include seasonal differencing to account for the seasonality present in the data.

Based on these initial results, we can set d = 1 for our initial model parameters. 

Let's now determine what D should be.


```python
fig, ax = plt.subplots(4,3, figsize=(9,9), sharex=True)
splotdata = data.copy()
splotdata['1st Difference'] = splotdata['Champagne Sales'] - splotdata['Champagne Sales'].shift(12)
splotdata['2nd Difference'] = splotdata['1st Difference'] - splotdata['1st Difference'].shift(12)
splotdata['3rd Difference'] = splotdata['2nd Difference'] - splotdata['2nd Difference'].shift(12)
splotdata['Champagne Sales'].plot(kind='line', ax=ax[0][0])
splotdata['Champagne Sales'].rolling(window=12).mean().plot(ax=ax[0][1])
splotdata['Champagne Sales'].rolling(window=12).std().plot(ax=ax[0][2])
splotdata['1st Difference'].plot(kind='line', ax=ax[1][0])
splotdata['1st Difference'].rolling(window=12).mean().plot(ax=ax[1][1])
splotdata['1st Difference'].rolling(window=12).std().plot(ax=ax[1][2])
splotdata['2nd Difference'].plot(kind='line', ax=ax[2][0])
splotdata['2nd Difference'].rolling(window=12).mean().plot(ax=ax[2][1])
splotdata['2nd Difference'].rolling(window=12).std().plot(ax=ax[2][2])
splotdata['3rd Difference'].plot(kind='line', ax=ax[3][0])
splotdata['3rd Difference'].rolling(window=12).mean().plot(ax=ax[3][1])
splotdata['3rd Difference'].rolling(window=12).std().plot(ax=ax[3][2])
ax[0][0].set_title('Time Series')
ax[0][1].set_title('Rolling Mean')
ax[0][2].set_title('Rolling Std Dev')
ax[0][0].set_ylabel('Original')
ax[1][0].set_ylabel('1st')
ax[2][0].set_ylabel('2nd')
ax[3][0].set_ylabel('3rd')
plt.show()
```


![png](/images/Time-Series-Forecasting-Models-ARIMA_files/Time-Series-Forecasting-Models-ARIMA_9_0.png)


Again, our data looks to become stationary after taking the second order seasonal difference. In practice, it is best to construct a number of models, trying out differenct p,d,q combiations to find the best performing model. We'll set our D term to 2 and move on.

Next, we need to check the autocorrelation and partial autocorrelation plots of the differenced time series to decide whether to include an AR or a MA component.


```python
acf_lag = stattools.acf(plotdata['2nd Difference'].iloc[2:], nlags=30)
pacf_lag = stattools.pacf(plotdata['2nd Difference'].iloc[2:], nlags=30, method='ols')

fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].stem(acf_lag)
ax[0].axhline(0, color='k')
ax[0].axhline(-1.96/np.sqrt(len(data)), linestyle='--',color='gray')
ax[0].axhline(1.96/np.sqrt(len(data)), linestyle='--',color='gray')

ax[1].stem(pacf_lag)
ax[1].axhline(0, color='k')
ax[1].axhline(-1.96/np.sqrt(len(data)), linestyle='--',color='gray')
ax[1].axhline(1.96/np.sqrt(len(data)), linestyle='--',color='gray')
plt.show()
```


![png](/images/Time-Series-Forecasting-Models-ARIMA_files/Time-Series-Forecasting-Models-ARIMA_11_0.png)


THere are several rules of thumb we use when looking at the autocorrelation and partial autocorrelation plots to determine whether to use AR or MA terms in the model, and what lag to set for the terms.

The first significant autocorrelation and partial autocorrelation terms are at a lag of 1, suggesting we set p and q to 1. There are significant autocorrelations at 12 and 24 suggesting the S should be 12. Since these are positive correlations, our first try to should be use the AR component and not the MA component for the seasonal part of the model, so we set P = 1 and Q = 0.

Now we have a model that looks like SARIMA(1,2,1)(1,2,0)[12]

### Build and validate the model

Now that we have determined the type of model to build, we can use statsmodels to do so. After we construct the model, we will compute performance metrics to describe the performance and allow us to compare the model to others we could build for this dataset.


```python
model = statespace.sarimax.SARIMAX(train, order=(1,2,1), seasonal_order=(1,2,0,12), enforce_invertibility=False)
model_fit = model.fit()
pred = model_fit.predict(start=test.index[0], end=test.index[-1])
```


```python
# calculate metrics

def equal_arrays(x1, x2):
    if len(x1) != len(x2):
        raise ValueError("The inpute arrays are not the same length")

def me(y_pred, y_test):
    equal_arrays(y_pred, y_test)
    n = len(y_test)
    return np.sum(y_pred - y_test)/n

def mpe(y_pred, y_test):
    equal_arrays(y_pred, y_test)
    n = len(y_test)
    return (np.sum((y_pred / y_test)-1)/n) * 100

def rmse(y_pred, y_test):
    equal_arrays(y_pred, y_test)
    n = len(y_test)
    return (np.sum((y_pred-y_test)**2)/n)**0.5

def mae(y_pred, y_test):
    equal_arrays(y_pred, y_test)
    n = len(y_test)
    return np.sum(abs(y_pred - y_test))/n

def mape(y_pred, y_test):
    equal_arrays(y_pred, y_test)
    n = len(y_test)
    return (np.sum(abs((y_pred/y_test)-1))/n) * 100

def mase(y_pred, y_test):
    equal_arrays(y_pred, y_test)
    n = len(y_test)
    performance = mae(y_pred, y_test)
    mean_diff = np.mean(abs(y_test - y_test.shift(1)))
    return performance / mean_diff   
```


```python
results = {'ME':me(pred, test), 'MPE':mpe(pred, test), 'RMSE':rmse(pred, test),
          'MAE':mae(pred, test), 'MAE':mae(pred, test), 'MAPE':mape(pred, test),
          'MASE':mase(pred, test), 'AIC':model_fit.aic}
```


```python
result_frame = pd.DataFrame(results ,index=['ARIMA(1,2,1)(1,2,0)[12]'])
result_frame
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
      <th>AIC</th>
      <th>MAE</th>
      <th>MAPE</th>
      <th>MASE</th>
      <th>ME</th>
      <th>MPE</th>
      <th>RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARIMA(1,2,1)(1,2,0)[12]</th>
      <td>1099.915212</td>
      <td>1782.259734</td>
      <td>34.314776</td>
      <td>0.637204</td>
      <td>1782.259734</td>
      <td>34.314776</td>
      <td>2048.576799</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(train)
plt.plot(test)
plt.plot(test.index, pred)
plt.legend(['Train','Test','Forecast'])
plt.show()
```


![png](/images/Time-Series-Forecasting-Models-ARIMA_files/Time-Series-Forecasting-Models-ARIMA_17_0.png)


Considering the performance metrics and the plot of the predictions compared to the test data, it looks like our model is over-estimating in its forecast. Notice the ME is much higher than 0. Furthermore, the MAPE and the MPE are the same, which would arrise when predictions are always greater than the actual data.

### Forecast

Now we can retrain our model on the full dataset and forecast future values of champagne sales.


```python
# predictions with holdout sample added back to the model
full_model = statespace.sarimax.SARIMAX(data, order=(1,2,1), seasonal_order=(1,2,0,12), enforce_invertibility=False)
full_model_fit = full_model.fit()
preds = full_model_fit.predict(start=96,end=102) # make 6 months of predictions
```


```python
# now calculate confidence intervals for new test x-series
mean_x = np.mean(data).values # mean of x
n = len(data) # number of samples in origional fit
t = 1.96 # t-value for 95% CI
y_err = (full_model_fit.predict(start=data.index[0], end=data.index[-1]) - data.iloc[:,0]) # residuals
s_err = np.sum(np.power(y_err,2)) # sum of the squares of the residuals
confs = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((preds-mean_x),2)/
        ((np.sum(np.power(data.values,2)))-n*(np.power(mean_x,2))))))
lower = preds - abs(confs)
upper = preds + abs(confs)
```


```python
# plot predictions with 95% CI
plt.plot(data)
plt.plot(preds, 'g')
plt.plot(preds.index, lower, color='r')
plt.plot(preds.index, upper, color='r')
plt.legend(['Actual','Forecast','95% CI'])
plt.ylabel('Sales ($)')
plt.title('2018 Champagne Sales Forecst')
plt.show()
```


![png](/images/Time-Series-Forecasting-Models-ARIMA_files/Time-Series-Forecasting-Models-ARIMA_21_0.png)


In this post, you have learned how to create and evaluate a seasonal ARIMA time series model to forecast the future! You should now be able to apply these concepts to new datasets. The model here could stand to be improved with further tuning of the 6 p, d, and q parameters. Can you best my results?

How do these results compare to the ETS model from my previous [post](https://johnkmaxi.github.io/blog/2018/07/25/Time-Series-Forecasting-Models-ETS)? Which model would you choose?
