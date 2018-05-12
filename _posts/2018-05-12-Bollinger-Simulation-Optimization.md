
# Can I Beat the Stock Market? A Simulation and Analysis

I have always been intrigued by the idea of being an active stock investor. The allure of developing a strategy to beat the market with some fancy algorithm while I travel the world with my family seems pretty great! To this end, I recently started the [Machine Learning for Trading](https://www.udacity.com/course/machine-learning-for-trading--ud501) course at Udacity. One of the concepts covered early on in this class are Bollinger Bands (BB). BBs use a 20-day rolling mean of the stock price to compute the an upper and lower price band indicating the price that is two standard deviations above or below the rolling mean. When the daily price exceeds these bands it indicates that the price is relatively high or low and is likely to regress towards the 20-day mean. I could see the riches in my mind's eye - it seems like such an easy way to execute the age old mantra of "buy low, sell high". When price dips below the lower band, buy, and when it rises past the upper band, sell. Easy, right? I decided to test this strategy strategy and which gave me the opportunity to practice using APIs to acces data, creating my own datasets, and building and importing my own modules. 

Let's get started!

First, we need to import some libraries. `quandlekey` is a module I created that includes functions to accesss the quandl API, such as my API key and function that grabs stock data from the WIKI database, and a class `BollingerBands` that implements my trading strategy. We will also use the pandas library for this simulation when it comes time to do that analysis. (I should also add that I use matplotlib, which is imported automatically when I open a jupter notebook. I set this in a config file.)


```python
import quandlkey
import pandas as pd
```

To test my startegy, we'll need to choose some stocks to work with. I created a stocks variable in `quandlkey` that has all the stock symbols available in the Quandl WIKI database. I want to test if my strategy, is stock agnostic, so we will randomly sample a set of stocks from the available list.


```python
# pick a list of stocks to use in the simulation
stocks = quandlkey.stocks()[0].sample(49).str[5:]
stocks
```




    2024     OHRP
    155       EQR
    1032     CNOB
    1537      IRG
    615       AFG
    1599      IRF
    410       SLM
    2640     TRST
    2243     ROLL
    542       ACM
    1019      CTG
    2632      TRN
    78        CAG
    2354      SWM
    339      ORLY
    808       BCC
    2460     STMP
    633      AMPE
    3133      ORM
    2332     SAFT
    505       AAN
    125       DHR
    1069       CR
    437       TSN
    606      UHAL
    2252     RCPT
    360       PFE
    489       YUM
    2189     PSMT
    977      CHCO
    1389       GB
    2918      ESS
    1952     NEON
    1511     HSNI
    1896     MOVE
    300       MON
    23        AFL
    1562    IMKTA
    2025      OIS
    1047      CPA
    1650      KAI
    927     CNBKA
    1302      FGL
    2882      AEC
    135       DOW
    533        AE
    973      CTRN
    2971     FULT
    2208     PTCT
    Name: 0, dtype: object



The simulations will be carried during a random time interval. However, we need to set a `start_date` which is the earliest timepoint we will consider during the simulation. The Machine Learning for Trading course discusses survivor bias in the stock market. This refers to the fact that most historical databases only include stocks/companies that still exist today. Thus, we are biased towards picking companies that have some moderate level of success. To mitigate this bias, let's pick a start date that occurs relatively recently. This way we can avoid picking stocks that made it through the 2008 crash. Those stocks are more likely to have dipped and seen growth since that time, which might artificially inflate our results. 2015 to know seems like a safe time frame to investigate.


```python
# pick a date range for which the stocks are available, perhaps use relatively recent tiem frame, 3-5 years
start_date = (2015, 1, 1)
```

The following code performs all the actual simulation. We create a few lists to accumulate the results, set the values we will try for the different parameters of the simulation and then loop through all combinations of the parameter to perform a grid search. For each stock, we try each combination of parameters on 10 different time periods. This way we can avoid biasing the results towards the performance of a particular company during the period since 2015. In order for the strategy to be successful, it needs to work on any stock at any time.


```python
# strategy
baseresults = []
myresults = []
params = []
company = []
starts = []
ends = []
windows = [2,5,10,20,25,30,40,50]
sell_pers = [0.1,0.25,0.50,0.75,0.90]
buy_amts = [50,100,250,500,750]
widths = [1,2,3]
initial_balance = 1000
sim_num = 0
for stock in stocks:
    stock_data = quandlkey.quandl_stocks(stock, start_date=start_date)
    try:
        for window in windows:
            for sell_per in sell_pers:
                for buy_amt in buy_amts:
                    for width in widths:
                        while sim_num < 10:
                            start = stock_data['WIKI/{} - Adj. Close'.format(stock)].sample(1).index[0]
                            end = stock_data['WIKI/{} - Adj. Close'.format(stock)][start:].sample(1).index[0]
                            sim = quandlkey.BollingerBands(stock_data['WIKI/{} - Adj. Close'.format(stock)].loc[start:end], 
                                           window=window, sell_per=sell_per, buy_amt=buy_amt, width=width,
                                            balance=initial_balance)
                            sim.trade()
                            base = sim.buy_and_hold()
                            baseresults.append(base[-1])
                            thisreturn = sim.results['Shares']*sim.stock+sim.results['Balance']
                            myresults.append(thisreturn[-1])
                            params.append((window, sell_per, buy_amt, width))
                            company.append(stock)
                            starts.append(start)
                            ends.append(end)
        
                            sim_num += 1
                        sim_num = 0
    except:
        pass
```

Now, let's save the data so we can access it later. We will first convert the list of tuples into a DataFrame, and then convert the remaing lists into DataFrame columns, concatenate them all together and save it in a csv. We will also convert the dollar amounts calculated into percent returns.


```python
# convert params list of tuples into a DataFrmae
params = pd.DataFrame(params, columns=['Window','SellPer','BuyAmt','Width'])
```


```python
paramSim = pd.DataFrame({'Base':baseresults,'Maxi':myresults, 'Company':company, 'Start':starts, 'End':ends})
paramSim = pd.concat([paramSim, params], axis=1)
# convert dollar returns to percentages
paramSim[['Base','Maxi']] = (paramSim[['Base','Maxi']]/initial_balance)-1 # converts dollar returns to % returns
```


```python
paramSim.to_csv('Bollinger Simulation Results.csv')
```

## Analysis

Now that we have the results, we need to analyze the data to gauge our level of success. Before begining any analysis, it is important to lay out the questions to be answered, otherwise it can be easy to get distracted during the exploratory analysis phase. Sttating your questions upfront and why they matter keeps the analysis focused. Questions can always be added to the list during exploratory analysis. Once the analysis begins, there will always be new questions to ask. The questions I want to answer going into this analysis are:

1. What was the average return for the two strategies?
    - Why: compares random stock investing to a strategy
2. Did any combination of parameters reliably beat the market?
    - Why: AAPL/GE analysis indicate that only a fraction of time my stategy beats the market. Do one of the other combinations work better? (whole point of doing this grid search)
3. Where there any specific companies that my strategy worked on?
    - Why: Would indicate that the strategy only works for certain types of companies.
    - Follow up to this result would be to test on other similar companies. Another project stemming from this would be to build an algorithm that could predict when my strategy would work.


```python
paramSim = pd.read_csv('Bollinger Simulation Results.csv', index_col=0)
```

## 1. What was the average return for the two strategies?
    - Why: compares random stock investing to a strategy


```python
# quick histogram, requires some serious tuning for good viz, the skew of the returns using my strategy is crazy
# tells us what type of distribution we are dealing with which informs which measure of central tendency to use
# for comparing the strategies
%matplotlib inline
fig, ax = plt.subplots(figsize=(16,9))
paramSim['Base'].plot(kind='hist', bins=50, ax=ax, legend=True, fontsize=20)
paramSim['Maxi'].plot(kind='hist', bins=50, ax=ax, legend=True, fontsize=20, alpha=0.5);
```


![png](/images/Bollinger-Simulation-Optimization_files/Bollinger-Simulation-Optimization_15_0.png)



```python
paramSim[['Base','Maxi']].describe()
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
      <th>Base</th>
      <th>Maxi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>288000.000000</td>
      <td>288000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.057131</td>
      <td>-0.179204</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.376709</td>
      <td>0.330758</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.978616</td>
      <td>-1.242240</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.085100</td>
      <td>-0.237009</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.006193</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.166588</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.972200</td>
      <td>4.249675</td>
    </tr>
  </tbody>
</table>
</div>



The figure above shows the distribution of the returns are roughly normal for the baseline strategy. The returns using my strategy are skewed in the negative direction. The quick stats shown by the describe function indicate that, overall, my strategy did not beat market, as both the mean and median are less than the baseline returns. Note that the values shown are decimal values, so to convert to a percent we need to multiply these numbers by 100. Because of the skew evident in the Maxi column, I think it is more appropriate to compare the median values. The median is a non-parametric measure of centrality. A non-parametric measure of dispersion to use in this case isthe inter-quartile range (IQR). The IQR is defined by the difference between the values at the 25th and 75th percentile. Numpy has a percentile function that makes this easy to calculate the IQR so we can display it on our plot.


```python
def iqr(x):
    """x 1D array/Series"""
    # calculate interquartile range
    q25, q75 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = q75 - q25
    return iqr
```

## Plot the median return


```python
%matplotlib inline
# calculate interquartile range
iqr_b = iqr(paramSim['Base'])
iqr_m = iqr(paramSim['Maxi'])
f, a = plt.subplots(figsize=(16,9))
paramSim[['Base','Maxi']].median().plot(kind='bar', yerr=[iqr_b,iqr_m], ax=a,
                                      fontsize=20, rot=0)
a.set_title('Median Investment Return', fontsize=20)
a.set_ylabel('% Return/100', fontsize=20)
a.set_xlabel('Strategy', fontsize=20)
a.text(0.1,.05,'{:.3f}%'.format(paramSim['Base'].median()*100), fontdict={'fontsize':20})
a.text(1.05,.05,'{:.3f}%'.format(paramSim['Maxi'].median()*100), fontdict={'fontsize':20});
```


![png](/images/Bollinger-Simulation-Optimization_files/Bollinger-Simulation-Optimization_20_0.png)


The median return values look pretty similar. And with a median return of 0%, there really isn't any incetive so far to put my strategy into practice. I'm also curious to see the mean return for each strategy. Typically, one can expect a 7% return on the stock market investments over the long-term. Since this benchmark is an average, it will provide us an external baseline for understanding the average performance of the stocks used in the simulation.

## Plot the mean return


```python
%matplotlib inline
f, a = plt.subplots(figsize=(16,9))
paramSim[['Base','Maxi']].mean().plot(kind='bar', yerr=paramSim[['Base','Maxi']].sem(), ax=a,
                                      fontsize=20, rot=0)
a.set_title('Mean Investment Return', fontsize=20)
a.set_ylabel('% Return/100', fontsize=20)
a.set_xlabel('Strategy', fontsize=20)
a.text(0-.05,.025,'{:.2f}%'.format(paramSim['Base'].mean()*100), fontdict={'fontsize':20})
a.text(.935,-0.12,'{:.2f}%'.format(paramSim['Maxi'].mean()*100), fontdict={'fontsize':20});
```


![png](/images/Bollinger-Simulation-Optimization_files/Bollinger-Simulation-Optimization_22_0.png)


The baseline strategy gained almost 6% on average, which is slightly below the expected 7%. I would expect this to get closer to 7% if we used a longer time frame. The investments in this simulation could be held for as little as one day or as long as three years. 

Since the distributions do not look normal, and the samples are matched, the most appropriate statistical test to compare the means is the [Wilcoxon Matched Pairs Test](http://www.biochemia-medica.com/content/comparing-groups-statistical-differences-how-choose-right-statistical-test). This test is a non-parametric version of the paired t-test. We are using the paired t-test because we care about the difference between the two strategie for each observation. Scipy has this test implemented and can imported to compute the statistic. However, the Scipy function does not return Z statistic. I have copied the Scipy source code and modified the function to return the Z statistic, which is used to compute Cohen's D. Cohen's D is statistic indicating [effect size](https://stats.stackexchange.com/questions/133077/effect-size-to-wilcoxon-signed-rank-test). It is important to calculate in this scenario because there are so many observations be used in the statistical test. With such a large N, the p-value is likely meaningless. Knowing the magnitude of the effect allows for us to say, "Sure, it was significant, but the effect size was so small, it doesn't matter." 


```python
from scipy.stats import wilcoxon
from scipy import stats
from scipy.stats import distributions
```


```python
def wilcoxon_(x, y=None, zero_method="wilcox", correction=False):
    """
    Calculate the Wilcoxon signed-rank test.
    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.
    Parameters
    ----------
    x : array_like
        The first set of measurements.
    y : array_like, optional
        The second set of measurements.  If `y` is not given, then the `x`
        array is considered to be the differences between the two sets of
        measurements.
    zero_method : string, {"pratt", "wilcox", "zsplit"}, optional
        "pratt":
            Pratt treatment: includes zero-differences in the ranking process
            (more conservative)
        "wilcox":
            Wilcox treatment: discards all zero-differences
        "zsplit":
            Zero rank split: just like Pratt, but spliting the zero rank
            between positive and negative ones
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic.  Default is False.
    Returns
    -------
    T : float
        The sum of the ranks of the differences above or below zero, whichever
        is smaller.
    p-value : float
        The two-sided p-value for the test.
    Notes
    -----
    Because the normal approximation is used for the calculations, the
    samples used should be large.  A typical rule is to require that
    n > 20.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    """

    if not zero_method in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method should be either 'wilcox' \
                          or 'pratt' or 'zsplit'")

    if y is None:
        d = x
    else:
        x, y = map(np.asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')
        d = x-y

    if zero_method == "wilcox":
        d = np.compress(np.not_equal(d, 0), d, axis=-1)  # Keep all non-zero differences

    count = len(d)
    if (count < 10):
        warnings.warn("Warning: sample size too small for normal approximation.")
    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r, axis=0)
    r_minus = np.sum((d < 0) * r, axis=0)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r, axis=0)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    T = min(r_plus, r_minus)
    mn = count*(count + 1.) * 0.25
    se = count*(count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        r = r[d != 0]

    replist, repnum = stats.find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = np.sqrt(se / 24)
    correction = 0.5 * int(bool(correction)) * np.sign(T - mn)
    z = (T - mn - correction) / se
    prob = 2. * distributions.norm.sf(abs(z))
    return T, prob, z
T, p, z = wilcoxon_(paramSim['Base'],paramSim['Maxi'])
print(T, p, z)
```

    10410007282.5 0.0 -231.195824242
    

## Compute the effect size


```python
print("Cohen's d: {:.2f}".format(z/np.sqrt(len(paramSim))))
```

    Cohen's d: -0.43
    

A Cohen's d with a magnitude of 0.43 is considered a moderate effect size. The negative value means that there was a decrease in the values in the Maxi strategy compared to the baseline strategy. These results indicate that for a given observation, the Maxi strategy gives a worse return then randomly investing in a particular stock. The next question we set out to answer was:

#### Did any combination of parameters reliably beat the market?
    - Why: AAPL/GE analysis indicate that only a fraction of time my stategy beats the market. Do one of the other combinations work better? (whole point of doing this grid search)
    
To answer this question, we can use the groupby function in pandas to group the returns for each combination of model parameters.

## Plot the returns for each parameter combination


```python
f, a = plt.subplots(figsize=(16,9))
parameter_groups = paramSim.groupby(['Width','SellPer','BuyAmt','Window'])[['Base','Maxi']].mean()
parameter_groups.plot(kind='bar', ax=a, yerr=paramSim.groupby(['Width','SellPer','BuyAmt','Window'])[['Base','Maxi']].sem());
```


![png](/images/Bollinger-Simulation-Optimization_files/Bollinger-Simulation-Optimization_30_0.png)


This plot gives a good overall picture of the fact that my strategy failed in the majority of cases. Since our interest is in the case in which my strategy actually did perform, we want to find the group combination that had the best results.

## Locate the best performing parameters


```python
best_return = parameter_groups['Maxi'].max()
best_return_index = parameter_groups[parameter_groups['Maxi']==best_return].index
```


```python
paramSim.groupby(['Width','SellPer','BuyAmt','Window'])[['Base','Maxi']].describe().loc[best_return_index,:].T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Width</th>
      <th>3</th>
    </tr>
    <tr>
      <th></th>
      <th>SellPer</th>
      <th>0.25</th>
    </tr>
    <tr>
      <th></th>
      <th>BuyAmt</th>
      <th>750</th>
    </tr>
    <tr>
      <th></th>
      <th>Window</th>
      <th>30</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">Base</th>
      <th>count</th>
      <td>480.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.077705</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.340341</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.883378</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.068422</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.029174</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.174757</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.332100</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Maxi</th>
      <th>count</th>
      <td>480.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.028492</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.169632</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.484250</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.864090</td>
    </tr>
  </tbody>
</table>
</div>




```python
paramSim[(paramSim['Width']==3)&(paramSim['SellPer']==0.25)&(paramSim['BuyAmt']==750)&(paramSim['Window']==30)]\
[['Base','Maxi']].hist(bins=100, figsize=(16,9));
```


![png](/images/Bollinger-Simulation-Optimization_files/Bollinger-Simulation-Optimization_34_0.png)


The descriptive statistics and the histograms illustrate that even when my strategy performed its best, it still didn't do as well as the market return for the same companies and time periods. Since these returns are matched, it would be informative to look at the difference in return from the two strategies, which leads to an additional question to consider in this analysis.

### Question to answer, what percent of the time does my strategy beat the market with best performing parameters?


```python
# first calculate a column representing the difference
paramSim['Diff'] = paramSim['Maxi'] - paramSim['Base']
```


```python
# now produce the descriptive statistics and histogram to get a general understanding of the data
grouped_diff = paramSim.groupby(['Width','SellPer','BuyAmt','Window'])['Diff']
grouped_diff.describe().loc[best_return_index,:].T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>Width</th>
      <th>3</th>
    </tr>
    <tr>
      <th>SellPer</th>
      <th>0.25</th>
    </tr>
    <tr>
      <th>BuyAmt</th>
      <th>750</th>
    </tr>
    <tr>
      <th>Window</th>
      <th>30</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>480.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.049213</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.293574</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.990400</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.153379</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.015460</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.080526</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.107390</td>
    </tr>
  </tbody>
</table>
</div>




```python
paramSim[(paramSim['Width']==3)&(paramSim['SellPer']==0.25)&(paramSim['BuyAmt']==750)&(paramSim['Window']==30)]\
['Diff'].hist(bins=100, figsize=(16,9));
```


![png](/images/Bollinger-Simulation-Optimization_files/Bollinger-Simulation-Optimization_38_0.png)


## How often Maxi beats Base in general


```python
len(paramSim[paramSim['Diff']>0])/len(paramSim)
```




    0.36681597222222223



In general, the difference between the strategies looks normally distributed. The risk of my strategy is apparant in the low rate of beating the market. I was only able to beat the market about 37% of the time. What if we consider only the most optimized strategy?

## What is the percentage of time that Maxi beats the market with optimized parameters?


```python
best_strategy = paramSim[(paramSim['Width']==3)&(paramSim['SellPer']==0.25)&(paramSim['BuyAmt']==750)&(paramSim['Window']==30)]
```


```python
print('% time beat the market: ', (len(best_strategy[best_strategy['Diff']>0]))/len(best_strategy))
print('Median return: ', best_strategy['Diff'].median())
```

    % time beat the market:  0.4708333333333333
    Median return:  -0.01546000000000003
    

## How often is my strategy profitable?


```python
print('% time strategy is profitable: ', (len(best_strategy[(best_strategy['Diff']>0)\
                                                            &(best_strategy['Maxi']>0)]))\
                                                              /len(best_strategy))
print((len(best_strategy[(best_strategy['Diff']>0)&(best_strategy['Maxi']>0)])))
```

    % time strategy is profitable:  0.10208333333333333
    49
    

Even with an optimized set of trading parameters, I still lose to the market more often than not. Only around 10% of the time am I able to turn a beat the market and turn a profit, which means close to 40% when I beat the market, I am still not profitable.

The next question is to look at different companies to further hone down on any situations in which my strategy might be applicable.


```python
best_by_company = best_strategy.groupby('Company')[['Maxi','Base','Diff']].mean()
when_it_worked = best_by_company[(best_by_company['Diff']>0)&(best_by_company['Maxi']>0)]
print(len(when_it_worked)/len(best_by_company))
print(when_it_worked.mean())
when_it_worked
```

    0.1875
    Maxi    0.090777
    Base    0.053194
    Diff    0.037584
    dtype: float64
    




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
      <th>Maxi</th>
      <th>Base</th>
      <th>Diff</th>
    </tr>
    <tr>
      <th>Company</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AE</th>
      <td>0.088235</td>
      <td>-0.007558</td>
      <td>0.095793</td>
    </tr>
    <tr>
      <th>BCC</th>
      <td>0.365406</td>
      <td>0.270545</td>
      <td>0.094861</td>
    </tr>
    <tr>
      <th>CTRN</th>
      <td>0.029352</td>
      <td>0.012415</td>
      <td>0.016936</td>
    </tr>
    <tr>
      <th>EQR</th>
      <td>0.021716</td>
      <td>0.015919</td>
      <td>0.005797</td>
    </tr>
    <tr>
      <th>ESS</th>
      <td>0.019269</td>
      <td>-0.012845</td>
      <td>0.032114</td>
    </tr>
    <tr>
      <th>HSNI</th>
      <td>0.003048</td>
      <td>-0.009217</td>
      <td>0.012265</td>
    </tr>
    <tr>
      <th>ORM</th>
      <td>0.021134</td>
      <td>0.020967</td>
      <td>0.000167</td>
    </tr>
    <tr>
      <th>TRN</th>
      <td>0.161243</td>
      <td>0.124428</td>
      <td>0.036814</td>
    </tr>
    <tr>
      <th>YUM</th>
      <td>0.107592</td>
      <td>0.064088</td>
      <td>0.043505</td>
    </tr>
  </tbody>
</table>
</div>



## I was able to beat the market in only 19% of the companies used in the simulation

Three of these companies had negative returns, four had modest levels of return, and two had high levels of return using a buy-and-hold startegy. On average, I was able to beat the market by almost 4%. Let's wrap up by picking one of the companies, and plotting the underlying data that generated the returns to see what stock market pattern allowed for my strategy to beat the market.


```python
best_strategy[best_strategy['Company']=='YUM']
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
      <th>Base</th>
      <th>Company</th>
      <th>End</th>
      <th>Maxi</th>
      <th>Start</th>
      <th>Window</th>
      <th>SellPer</th>
      <th>BuyAmt</th>
      <th>Width</th>
      <th>Diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>166040</th>
      <td>-0.142628</td>
      <td>YUM</td>
      <td>2015-09-24</td>
      <td>0.000000</td>
      <td>2015-07-20</td>
      <td>30</td>
      <td>0.25</td>
      <td>750</td>
      <td>3</td>
      <td>0.142628</td>
    </tr>
    <tr>
      <th>166041</th>
      <td>-0.162555</td>
      <td>YUM</td>
      <td>2016-01-06</td>
      <td>-0.164613</td>
      <td>2015-04-29</td>
      <td>30</td>
      <td>0.25</td>
      <td>750</td>
      <td>3</td>
      <td>-0.002058</td>
    </tr>
    <tr>
      <th>166042</th>
      <td>-0.047400</td>
      <td>YUM</td>
      <td>2016-12-08</td>
      <td>0.000000</td>
      <td>2016-08-29</td>
      <td>30</td>
      <td>0.25</td>
      <td>750</td>
      <td>3</td>
      <td>0.047400</td>
    </tr>
    <tr>
      <th>166043</th>
      <td>0.073335</td>
      <td>YUM</td>
      <td>2017-02-21</td>
      <td>0.416751</td>
      <td>2015-07-10</td>
      <td>30</td>
      <td>0.25</td>
      <td>750</td>
      <td>3</td>
      <td>0.343416</td>
    </tr>
    <tr>
      <th>166044</th>
      <td>0.008083</td>
      <td>YUM</td>
      <td>2016-08-17</td>
      <td>0.164855</td>
      <td>2015-06-29</td>
      <td>30</td>
      <td>0.25</td>
      <td>750</td>
      <td>3</td>
      <td>0.156772</td>
    </tr>
    <tr>
      <th>166045</th>
      <td>0.489000</td>
      <td>YUM</td>
      <td>2017-10-31</td>
      <td>0.155998</td>
      <td>2015-02-03</td>
      <td>30</td>
      <td>0.25</td>
      <td>750</td>
      <td>3</td>
      <td>-0.333002</td>
    </tr>
    <tr>
      <th>166046</th>
      <td>0.311840</td>
      <td>YUM</td>
      <td>2017-12-12</td>
      <td>0.168588</td>
      <td>2015-05-11</td>
      <td>30</td>
      <td>0.25</td>
      <td>750</td>
      <td>3</td>
      <td>-0.143252</td>
    </tr>
    <tr>
      <th>166047</th>
      <td>0.002000</td>
      <td>YUM</td>
      <td>2018-03-16</td>
      <td>0.000000</td>
      <td>2018-02-07</td>
      <td>30</td>
      <td>0.25</td>
      <td>750</td>
      <td>3</td>
      <td>-0.002000</td>
    </tr>
    <tr>
      <th>166048</th>
      <td>0.098652</td>
      <td>YUM</td>
      <td>2017-07-10</td>
      <td>0.000000</td>
      <td>2017-04-20</td>
      <td>30</td>
      <td>0.25</td>
      <td>750</td>
      <td>3</td>
      <td>-0.098652</td>
    </tr>
    <tr>
      <th>166049</th>
      <td>0.010550</td>
      <td>YUM</td>
      <td>2016-07-22</td>
      <td>0.334346</td>
      <td>2015-08-10</td>
      <td>30</td>
      <td>0.25</td>
      <td>750</td>
      <td>3</td>
      <td>0.323796</td>
    </tr>
  </tbody>
</table>
</div>




```python
symbol = 'YUM'
start = (2015,7,20)
stock = quandlkey.quandl_stocks(symbol, start_date=start)
sim = quandlkey.BollingerBands(stock['WIKI/YUM - Adj. Close'],window=30, sell_per=0.25, buy_amt=7550, width=3,
                                            balance=1000)
sim.trade()
base = sim.buy_and_hold()
```

## Graph of each simulated epoch

Each graph is normalized to start from 1. Text indicates the difference in percent profit, of my strategy versus the market.


```python
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(10, figsize=(16,9))
yum = best_strategy[best_strategy['Company']=='YUM']
yum['Start'] = pd.to_datetime(yum['Start'])
yum['End'] = pd.to_datetime(yum['End'])
for i, row in enumerate(yum.index):
    s = yum.loc[row,'Start']
    e = yum.loc[row,'End']
    ax[i].plot(stock.loc[s:e].index, base.loc[s:e]/base.iloc[0], label='{}'.format(str(s)), linewidth=3)
    ax[i].text(.5,.5,'{:.2f}'.format(yum.loc[row,'Diff']), transform=ax[i].transAxes, fontsize=15);
fig.tight_layout()
```

    c:\users\john maxi\anaconda3\envs\tensorflowenv\lib\site-packages\ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      after removing the cwd from sys.path.
    c:\users\john maxi\anaconda3\envs\tensorflowenv\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    


![png](/images/Bollinger-Simulation-Optimization_files/Bollinger-Simulation-Optimization_53_1.png)


## Profitable events had sudden drops in price followed by steadily rising prices

Plots 1, 3, 4, and 10 show sudden price drops early in the time period. The stock price then continued to rise after that initial drop. When the initial drop did not happen, my startegy was less profitable then the market. While my implementation of the strategy catches the low points in the market, it does not protect against selling for less than I bought for, or buying for less than I last sold for. A single big market event can make or break the profitability.

# Conclusions

- Bollinger Band-style trading does not incorporate a global view of how the stock is moving
- Bollinger Band-style trading has high risk and as implemented here is not likely to surpass a buy-and-hold strategy, **even when randomly choosing a stock**.
- Bollinger Bands are effective for identifying events when a stock is undervalued
