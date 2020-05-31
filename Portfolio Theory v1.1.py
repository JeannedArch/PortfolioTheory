#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
from pandas_datareader import data, wb

#plt.style.use('fivethirtyeight')
#np.random.seed(777)

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import os
os.getcwd()
os.chdir("/Users/geralddeutsch/desktop")


# In[88]:


import bs4 as bs
import datetime as dt
import os
from pandas_datareader import data as pdr
import pickle
import requests

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('.', '-')
        ticker = ticker[:-1]
        tickers.append(ticker) 
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers

tickers = save_sp500_tickers()
print(tickers)
print(len(tickers))
list_of_lists_tickers = [tickers[i:i+5] for i in range(0, len(tickers), 5)]


# In[4]:


newdf=pd.DataFrame()
for i in list_of_lists_tickers:
    for item in i:
        stock = data.DataReader(item, 
                           start='2019-5-4', 
                           end='2020-5-4', 
                           data_source='yahoo')['Adj Close']
        newdf=pd.concat([newdf,stock.rename(item)],axis=1,sort=False)
days=newdf.count()   
df_clean=newdf.T.drop_duplicates().T
print("Adjusted closing prices for ", len(newdf), " S&P500 companies")
print(df_clean)
pctg_returns = df_clean.pct_change().iloc[1:-1].copy()
mean_returns=pctg_returns.mean()
print("Daily mean returns are: ")
print(mean_returns)


# In[128]:


def portfolio_annualised_performance(weights, returns, cov_matrix):
    returns = np.sum(returns*weights)*252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, asset_number):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    names_record =[]
    for i in range(num_portfolios):
        mean_returns_sample = mean_returns.sample(asset_number, axis=0)
        sample_assets = mean_returns_sample.index.tolist()
        cov_matrix_loop = build_cov(sample_assets)
        #weight_df = pd.DataFrame(columns=df_clean.columns)
        #weight_df.append(mean_returns_sample, ignore_index=True)
        #weights = np.random.random(asset_number) #np.random.uniform(-.2,.2,numberstocks)
        weights = np.random.uniform(-1,1,asset_number)
        weights /= np.sum(weights)
        weights_record.append(weights)
        for i in range(asset_number):
            if weights[i]<0:
                mean_returns_sample[i] = mean_returns_sample[i]*-1 #INVERT RETURNS FOR NEGATIVE PORTFOLIO ALLOCATION
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns_sample, cov_matrix_loop)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        names_record.append(sample_assets)
    #results = results[~np.isnan(results).any(axis=1)]
    return results, weights_record,names_record

def build_cov(sample_assets):
    return np.cov(pctg_returns[sample_assets].T)


asset_number=30
#numberstocks = len(mean_returns)
#returns = pctg_returns
#mean_returns = mean_returns
cov_matrix = np.cov(mean_returns)
num_portfolios = 50000
risk_free_rate = 0.01

random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, asset_number)


# In[129]:


def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights, names_record = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate, asset_number)
    
    results = results[:, ~np.isnan(results).any(axis=0)]
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=names_record[max_sharpe_idx],columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*1,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=names_record[min_vol_idx],columns=['allocation'])
    min_vol_allocation.allocation = [round(i*1,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print ("-"*80)
    print ("Maximum Sharpe Ratio Portfolio Allocation\n")
    print ("Annualised Return:", round(rp,4))
    print ("Annualised Volatility:", round(sdp,4))
    print ("Sharpe ratio:", results[2,max_sharpe_idx])
    print ("\n")
    print (max_sharpe_allocation)
    print ("-"*80)
    print ("Minimum Volatility Portfolio Allocation\n")
    print ("Annualised Return:", round(rp_min,2))
    print ("Annualised Volatility:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='y',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    
display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)


# ## Next Steps
# - Constraint on max position size (and minimum size)
# - Possibility of short sales
# - Optionality of cash
# 
# - Import xls my positions from xls file and return individual 10 day, 30 day and annual volatility, and sharpe ratio
# - Display my portfolio on the Efficient Frontier 
# 
# Next level:
# - No fractional shares
# - Maximum number of positions
# 
# If short sale, sign of return needs to be changed

# # Code Schnippsel

# In[4]:


from pandas_datareader import data, wb
df=pd.DataFrame()
tickers = stickers
for i in tickers:
    a=tickers[0:26]
    for item in a:
        stock = data.DataReader(item, 
                           start='2019-5-4', 
                           end='2020-5-4', 
                           data_source='yahoo')['Adj Close']
        df=pd.concat([df,stock.rename(item)],axis=1,sort=False)
    del tickers[0:25]
#df.columns = stickers[0:26]
print(df)    


# In[12]:


df_clean=df.T.drop_duplicates().T
print(df_clean)


# In[6]:


df.to_csv (r'/Users/geralddeutsch/desktop/export_dataframe.csv', index = True, header=True)


# In[ ]:


i = 0
a = mean_returns.sample(30, axis=0)
weight_df = pd.DataFrame(columns=df_clean.columns)
weight_df.append(a, ignore_index=True)


#b=a.index.tolist()
#b=pd.Series(b)

#print(b)
#print("Type of b is ",type(b))
#newdf=pd.DataFrame()
#xxx = pd.concat([newdf,b],axis=1,sort=False)
#print(xxx)


# In[ ]:


results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate, asset_number)

max_sharpe_idx = np.argmax(results[2])
max_sharpe_idx
results
weights[19].


# In[ ]:


from pandas_datareader import data, wb
df=pd.DataFrame()

for i in tickers:
    a=tickers[0:25]
    for item in a:
        stock = data.DataReader(item, 
                           start='2019-5-4', 
                           end='2020-5-4', 
                           data_source='yahoo')['Adj Close']
        df=pd.concat([df,stock.rename(item)],axis=1,sort=False)
    del tickers[0:25]
    
df_clean=df.T.drop_duplicates().T
print("Clean Adjusted Prices for ", len(df), " S&P500 companies")
print(df_clean)


# In[104]:


weights = np.random.uniform(-1,1,30)
print(weights)
weights /= np.sum(weights)
print (weights)
print (np.sum(weights))


# In[105]:


1+1


# In[106]:


mean_returns_sample = mean_returns.sample(asset_number, axis=0)
sample_assets = mean_returns_sample.index.tolist()
cov_matrix_loop = build_cov(sample_assets)
#weights = np.random.random(asset_number) 
weights = np.random.uniform(-.1,.1,asset_number)
weights /= np.sum(weights)
print(type(weights),weights.shape)
for i in range(len(mean_returns_sample)+1):
    if weights[i]<0:
        mean_returns_sample[i]*-1
print(mean_returns_sample)


# In[126]:


mean_returns_sample = mean_returns.sample(asset_number, axis=0)
print(mean_returns_sample.shape)
sample_assets = mean_returns_sample.index.tolist()
cov_matrix_loop = build_cov(sample_assets)
#print(cov_matrix_loop)
print(mean_returns_sample)
weights = np.random.uniform(-1,1,asset_number)
weights /= np.sum(weights)
print(type(weights))
print("Weights are", weights)
print(weights.shape)
for i in range(asset_number):
    if weights[i]<0:
        mean_returns_sample[i] = mean_returns_sample[i]*-1
    #else:
print(mean_returns_sample)


# In[127]:


def portfolio_annualised_performance(weights, returns, cov_matrix):
    returns = np.sum(returns*weights)*252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

weights = weights
returns = mean_returns_sample
cov_matrix = cov_matrix_loop

portfolio_annualised_performance(weights, returns, cov_matrix)


# In[ ]:




