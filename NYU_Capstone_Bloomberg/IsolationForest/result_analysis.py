#%% Original Detection Plot
from detection_engine import Surpriver, ArgChecker
import pandas as pd
import datetime
def model(Start_time = '2024-08-01',end_time = '2024-08-31'):
    argumentChecker = ArgChecker()
    supriver = Surpriver(START=Start_time,END = end_time)
    return supriver
def Run_daily_Result(supriver):
    result = supriver.find_anomalies()
    name = []
    score = []
    date1 =[]
    for i in result:
        name.append(i[1])
        score.append(i[0])
        date1.append(i[2]['Datetime'].iloc[-1].date().strftime('%Y-%m-%d')
    )
    result_1 = pd.DataFrame()
    result_1['symbol'] = name
    result_1['score'] = score
    result_1['Date'] = date1
    result_1['anomaly'] = result_1['score']<0
    return result_1

# %% Generate Running days and train the model
from datetime import datetime, timedelta
import pandas as pd
period_days = 30
fit_start_date = datetime.strptime('2024-08-01', '%Y-%m-%d')
num_days = 30
date_list = [(fit_start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]

start_times = []
period_data = []
for start_i in date_list:
    start_date = datetime.strptime(start_i, '%Y-%m-%d')
    period_days = period_days
    end_date = start_date + timedelta(days=period_days)
    end_i = end_date.strftime('%Y-%m-%d')
    model_1= model(Start_time = start_i ,end_time = end_i)
    res = Run_daily_Result(model_1)
    period_data.append(res)
df = pd.concat([s for s in period_data], ignore_index=True)
df
# %% Single stock's historical performance
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
stock_prices = yf.download(tickers = 'LLY',start= '2024-08-01',
								end = '2024-10-11')
returns_price = stock_prices[['Adj Close']].pct_change().reset_index().rename(columns = {'Adj Close':'Return'}).dropna()
returns_price = returns_price[(returns_price.Date>='2024-08-30')&(returns_price.Date<='2024-09-27')]
df_aapl = df[df.symbol=='LLY']
df_aapl['Date'] = pd.to_datetime(df_aapl['Date'])
returns_price['Date'] = pd.to_datetime(returns_price['Date'])
plot_data = pd.merge(df_aapl,returns_price,on='Date',how='inner').set_index('Date')
class TimeSeriesPlotter:
    def __init__(self, display_returns: pd.Series, anomalies: pd.Series):
        self.display_returns = display_returns
        self.anomalies = anomalies

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.display_returns.index, self.display_returns.values, label='Returns', color='blue')
        anomaly_dates = self.display_returns.index[self.anomalies]
        anomaly_values = self.display_returns[self.anomalies]
        ax.scatter(anomaly_dates, anomaly_values, color='red', label='Anomalies', marker='o')
        ax.set_title('Time Series of Returns with Anomalies')
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns')
        ax.legend()
        plt.show()
plotter = TimeSeriesPlotter(plot_data['Return'], plot_data['anomaly'])
plotter.plot()
# %% Features Importance Plot(single day prediction)
def Run_daily_Result_performance(Start_time = '2024-08-01',end_time = '2024-08-31'):
    supriver = model(Start_time,end_time)
    supriver.find_anomalies()
    return supriver.calculate_shape()
Run_daily_Result_performance()
# %%
