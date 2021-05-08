#投資組合優化
#擁有最大夏普比率的投資組合分配就是最佳的組合
#以兩種方式尋找:(1)隨機猜測(隨機猜測千萬組不同的分配)(2)數學優化

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ibm= pd.read_csv('D:\\IBM_CLOSE.csv', index_col='Date', parse_dates=True)
intc= pd.read_csv('D:\\INTC_CLOSE.csv', index_col='Date', parse_dates=True)
msft= pd.read_csv('D:\\MSFT_CLOSE.csv', index_col='Date', parse_dates=True)
amzn= pd.read_csv('D:\\amzn_CLOSE.CSV', index_col='Date', parse_dates=True)
stocks =pd.concat([ibm, intc, msft, amzn], axis=1)
stocks.columns= ['ibm', 'intel', 'microsoft', 'amazon']
print(stocks.head())

mean_daily_ret= stocks.pct_change(1).mean()
print(mean_daily_ret)
print(stocks.pct_change(1).corr()) #股票之間的相連性
stock_normed = stocks/stocks.iloc[0] #歸一化回報
stock_normed.plot()
#plt.show()
#現在切換到log returns，大多數技術分析需要對時間序列進行去除或標準化
log_re = np.log(stocks/stocks.shift(1))
print(log_re.head())
log_re.hist(bins=100, figsize=(12,8))
plt.tight_layout()
#plt.show()
a =log_re.describe().transpose()
pd.set_option('max_columns',8)
pd.set_option('max_rows',4)
print(a)

log_re.mean()*252
print(log_re.cov()) #計算他們之間的協方差(稍後用於計算投資組合的波動率)
log_re.cov()*252
#單次運行的隨機分配
np.random.seed(101)
print('股票')
print(stocks.columns)
print('\n')
print('創建隨機權重')
weights=np.array(np.random.random(4))
print(weights)
print('\n')
print('重新平衡權重')
weights= weights/np.sum(weights)
print(weights)
print('\n')
print('預期的投資組合回報')
exp_re= np.sum(log_re.mean()*weights)*252
print(exp_re)
print('\n')
print('預期波動率')
exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_re.cov()*252, weights)))
print(exp_vol)
print('\n')
print('夏普比率')
SR=exp_re/exp_vol
print(SR)
#嘗試使用蒙地卡羅模擬隨機找到最佳投資組合，在15000個分配權重中，找到最大夏普比率
num_times= 15000
all_weights= np.zeros((num_times, len(stocks.columns)))
re_arr = np.zeros(num_times)
vol_arr = np.zeros(num_times)
sharpe_arr= np.zeros(num_times)

for idx in range(num_times):
	weights= np.array(np.random.random(4))
	weights= weights/np.sum(weights)
	all_weights[idx,:]= weights
	re_arr[idx]= np.sum((log_re.mean()*weights)*252)
	vol_arr[idx]= np.sqrt(np.dot(weights.T, np.dot(log_re.cov()*252, weights)))
	sharpe_arr[idx]= re_arr[idx]/vol_arr[idx]
print(sharpe_arr.max())
print(sharpe_arr.argmax())
print(all_weights[11710,:])
#儲存回報與波動率
max_sr_re= re_arr[11710]
max_sr_vol= vol_arr[11710]

plt.figure(figsize=(12,8))
plt.scatter(vol_arr, re_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel=('Volatility')
plt.ylabel=('Return')
plt.scatter(max_sr_vol, max_sr_re, c='red', s=50, edgecolors='black')
#plt.show()

###數學優化
def get_re_vol_sr(weights):
	weights= np.array(weights)
	exp_re = np.sum(log_re.mean()*weights)*252
	exp_vol= np.sqrt(np.dot(weights.T, np.dot(log_re.cov() * 252, weights)))
	SR= exp_re/exp_vol
	return np.array([exp_re, exp_vol, SR])
from scipy.optimize import minimize
def neg_sr(weights):
	return get_re_vol_sr(weights)[2]* -1
def check_sum(weights):
	return np.sum(weights)-1
cons=({'type':'eq', 'fun':check_sum})
bounds= ((0,1),(0,1),(0,1),(0,1))
init_guess= [0.25, 0.25, 0.25, 0.25]
opt_results= minimize(neg_sr, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
print(opt_results)
print(get_re_vol_sr(opt_results.x)) #回報、波動率、夏普比率
##有效邊界
frontier_y= np.linspace(0,0.3,100)
def minimize_volatility(weights):
	return get_re_vol_sr(weights)[1]
frontier_volatility= []
for possible_return in frontier_y:
	cons= ({'type':'eq', 'fun':check_sum},
		{'type':'eq', 'fun':lambda w: get_re_vol_sr(w)[0]-possible_return})
	result = minimize(minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
	frontier_volatility.append(result['fun'])
plt.figure(figsize=(12,8))
plt.scatter(vol_arr, re_arr, c= sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel='Volatility'
plt.ylabel='Return'
plt.plot(frontier_volatility, frontier_y, 'r--', linewidth=3)
plt.show()



