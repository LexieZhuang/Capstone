
from detection_engine import Surpriver, ArgChecker
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf

class Result_analysis:
    def __init__(self, start_date, num_days, period_days):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.num_days = num_days
        self.period_days = period_days
        self.results = []

    def create_model(self, start_time, end_time):
        self.model = Surpriver(START=start_time, END=end_time)
        return self.model

    # calculate anonaly scores
    def process_results(self, raw_results):
        name, score, date1 = [], [], []
        for item in raw_results:
            name.append(item[1])
            score.append(item[0])
            date1.append(item[2]['Datetime'].iloc[-1].date().strftime('%Y-%m-%d'))
        result_df = pd.DataFrame({
            'symbol': name,
            'score': score,
            'Date': date1
        })
        result_df['anomaly'] = result_df['score'] < 0
        return result_df

    # run detection for multiple time slots
    def run_detection(self):
        date_list = [(self.start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(self.num_days)]
        for start_i in date_list:
            end_date = datetime.strptime(start_i, '%Y-%m-%d') + timedelta(days=self.period_days)
            end_i = end_date.strftime('%Y-%m-%d')
            self.create_model(start_i, end_i)
            raw_results = self.model.find_anomalies()
            processed_data = self.process_results(raw_results)
            self.results.append(processed_data)
        combined_data = pd.concat(self.results, ignore_index=True)
        self.df = combined_data
        return combined_data
    
    # shap plot
    def Run_daily_Result_performance(self,time = '2024-08-01'):
        end_time = pd.to_datetime(time) +pd.Timedelta(days = self.period_days)
        model = Surpriver(START=time, END=end_time)
        model.find_anomalies()
        return model.calculate_shape()
    
    # single stock's anomaly and return plot
    def Single_stock_plot(self,ticker_name='AAPL'):
        start_date = self.start_date - pd.Timedelta(days= 3)
        end_date = self.start_date + pd.Timedelta(days= self.num_days+self.period_days +1)
        stock_prices = yf.download(tickers = ticker_name,start=start_date ,end= end_date)
        returns_price = stock_prices[['Adj Close']].pct_change().reset_index().rename(columns = {'Adj Close':'Return'}).dropna()
        returns_price
        returns_price = returns_price[(returns_price.Date>= self.start_date)&(returns_price.Date<= end_date)]
        df = self.df
        df_aapl = df[df.symbol==ticker_name]
        df_aapl['Date'] = pd.to_datetime(df_aapl['Date'])
        returns_price['Date'] = pd.to_datetime(returns_price['Date'])
        plot_data = pd.merge(df_aapl,returns_price,on='Date',how='left').set_index('Date')
        plt.plot(figsize=(10, 6))
        plt.plot(plot_data['Return'].index, plot_data['Return'].values, label='Returns', color='blue')
        anomaly_dates = plot_data['Return'].index[plot_data['anomaly']]
        anomaly_values = plot_data['Return'][plot_data['anomaly']]
        plt.scatter(anomaly_dates, anomaly_values, color='red', label='Anomalies', marker='o')
        plt.title('Time Series of Returns with Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.legend()
        plt.show()

    # anomaly scores(anomaly score with absolute price change value)
    def anomaly_plot(self):
        return self.model.calculate_future_stats()

if __name__ == "__main__":
    example1 = Result_analysis('2024-08-01', 30, 30)
    final_results = example1.run_detection()
    example1.Run_daily_Result_performance()
    example1.Single_stock_plot()
    example1.anomaly_plot()

