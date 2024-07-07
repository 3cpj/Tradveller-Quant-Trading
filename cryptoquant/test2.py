import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging
import colorlog
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# 设置日志记录
def setup_logging():
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(f"%(log_color)s%(asctime)s - %(levelname)s - %(message)s"))
    file_handler = TimedRotatingFileHandler("quant_factors.log", when="h", backupCount=50)
    log_level = logging.INFO
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=log_level, handlers=[handler, file_handler])

setup_logging()

class FactorDiscoveryAgent:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.load_data()
        self.factors = pd.DataFrame(index=self.data.index)
        
    def load_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if data.empty:
            raise ValueError(f"No data available for {self.ticker} between {self.start_date} and {self.end_date}")
        logging.info(f"Loaded data shape: {data.shape}")
        return data

    def add_sma_factor(self, window):
        self.factors[f'sma_{window}'] = self.data['Close'].rolling(window).mean()

    def add_std_factor(self, window):
        self.factors[f'std_{window}'] = self.data['Close'].rolling(window).std()

    def add_momentum_factor(self, window):
        self.factors[f'momentum_{window}'] = self.data['Close'].diff(window)

    def generate_factors(self):
        windows = [5, 10, 20, 50, 100]
        for window in windows:
            self.add_sma_factor(window)
            self.add_std_factor(window)
            self.add_momentum_factor(window)
    
        # 删除所有包含 NaN 的行
        self.factors = self.factors.dropna()
        logging.info(f"Generated factors shape: {self.factors.shape}")

    def evaluate_factors(self):
        
        returns = self.data['Close'].pct_change().shift(-1)
        valid_factors = self.factors.dropna()
        valid_returns = returns.loc[valid_factors.index]

        logging.info(f"Shape of valid_factors: {valid_factors.shape}")
        logging.info(f"Shape of valid_returns: {valid_returns.shape}")
        logging.info(f"Number of NaN in valid_returns: {valid_returns.isna().sum()}")

        # 删除 NaN 值
        mask = ~np.isnan(valid_returns)
        X = valid_factors.values[mask]
        y = valid_returns.values[mask]

        logging.info(f"Shape of X after removing NaNs: {X.shape}")
        logging.info(f"Shape of y after removing NaNs: {y.shape}")

        if len(y) == 0:
            logging.warning("No valid data points after removing NaNs")
            return float('inf')

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        
        logging.info(f"Evaluated factors with MSE: {mse}")
        return mse

    def optimize_factors(self):
        best_mse = float('inf')
        best_factors = None

        for i in range(100):  # 100次迭代
            self.generate_factors()
            if self.factors.empty:
                logging.warning(f"Iteration {i}: No valid factors generated")
                continue
            
            try:
                mse = self.evaluate_factors()
                if mse < best_mse:
                    best_mse = mse
                    best_factors = self.factors.copy()
            except Exception as e:
                logging.error(f"Error in iteration {i}: {str(e)}")
        
        if best_factors is None:
            logging.error("Failed to find any valid factor combination")
            return

        self.factors = best_factors
        logging.info(f"Optimized factors with best MSE: {best_mse}")

    def run(self):
        self.optimize_factors()
        self.save_factors()
        self.plot_factor_correlation()
        self.plot_factor_importance()
        sharpe_ratio = self.backtest()
        self.plot_rolling_sharpe_ratio()
        logging.info(f"Final Sharpe Ratio: {sharpe_ratio}")

    def save_factors(self):
        self.factors.to_csv("optimized_factors.csv")
        logging.info(f"Saved optimized factors to optimized_factors.csv")

    def backtest(self):
        # 假设初始资本为1000000
        initial_capital = 1000000
        position = 0  # 当前持仓
        buy_signals = []
        sell_signals = []
        returns = []

        for i in range(1, len(self.factors)):
            row = self.factors.iloc[i]
            prev_row = self.factors.iloc[i-1]

            # 简单交易策略：如果当前SMA小于STD并且前一条记录的SMA大于等于STD，则买入
            if row['sma_20'] < row['std_40'] and prev_row['sma_20'] >= prev_row['std_40']:
                buy_signals.append(self.data.index[i])
                position = initial_capital / self.data['Close'][i]
                initial_capital = 0  # 全部资金买入

            # 如果当前SMA大于等于STD并且前一条记录的SMA小于STD，则卖出
            elif row['sma_20'] >= row['std_40'] and prev_row['sma_20'] < prev_row['std_40']:
                sell_signals.append(self.data.index[i])
                initial_capital = position * self.data['Close'][i]
                position = 0  # 全部卖出

            # 记录每日收益
            daily_return = (position * self.data['Close'][i] + initial_capital) - initial_capital
            returns.append(daily_return)

        self.data['Strategy Returns'] = returns

        # 计算夏普比率
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

        # 绘制收益曲线和交易信号
        self.plot_backtest(buy_signals, sell_signals)

        logging.info(f"Sharpe Ratio: {sharpe_ratio}")

        return sharpe_ratio

    def plot_backtest(self, buy_signals, sell_signals):
        plt.figure(figsize=(14, 7))

        plt.plot(self.data.index, self.data['Close'], label='BTC-USD Price')
        plt.plot(self.data.index, self.data['Strategy Returns'].cumsum(), label='Cumulative Strategy Returns', alpha=0.7)

        plt.scatter(buy_signals, self.data.loc[buy_signals]['Close'], marker='^', color='g', label='Buy Signal', s=100)
        plt.scatter(sell_signals, self.data.loc[sell_signals]['Close'], marker='v', color='r', label='Sell Signal', s=100)

        plt.title('BTC-USD Price and Cumulative Strategy Returns')
        plt.xlabel('Date')
        plt.ylabel('Price / Returns')
        plt.legend()
        plt.grid(True)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.close()
        logging.info("Backtest results plot saved as backtest_results.png")
        
    def plot_factor_correlation(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.factors.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Factor Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('factor_correlation.png')
        plt.close()
        logging.info("Factor correlation heatmap saved as factor_correlation.png")

    def plot_factor_importance(self):
        returns = self.data['Close'].pct_change().shift(-1)
        valid_data = pd.concat([self.factors, returns], axis=1).dropna()
        X = valid_data[self.factors.columns]
        y = valid_data[returns.name]
        
        importance = mutual_info_regression(X, y)
        importance_df = pd.DataFrame({'factor': X.columns, 'importance': importance})
        importance_df = importance_df.sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='factor', data=importance_df)
        plt.title('Factor Importance')
        plt.tight_layout()
        plt.savefig('factor_importance.png')
        plt.close()
        logging.info("Factor importance plot saved as factor_importance.png")

    def plot_rolling_sharpe_ratio(self, window=252):
        returns = self.data['Strategy Returns'].dropna()
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)

        plt.figure(figsize=(14, 7))
        plt.plot(rolling_sharpe.index, rolling_sharpe.values)
        plt.title(f'Rolling Sharpe Ratio (Window: {window} days)')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('rolling_sharpe_ratio.png')
        plt.close()
        logging.info(f"Rolling Sharpe Ratio plot saved as rolling_sharpe_ratio.png")

# 使用示例
if __name__ == "__main__":
    try:
        agent = FactorDiscoveryAgent(ticker='BTC-USD', start_date='2020-01-01', end_date='2024-01-01')
        agent.run()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
