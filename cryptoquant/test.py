import math
import numpy as np
import asyncio
import logging
import colorlog
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone, timedelta
import pandas as pd

# 定义日志记录
def setup_logging():
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}"))
    file_handler = TimedRotatingFileHandler("btc_netflow_1d.log", when="h", backupCount=50)
    log_level = logging.INFO
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
    logging.basicConfig(level=log_level, handlers=[handler, file_handler])

setup_logging()

class Strategy:
    # Indicator params
    holding_time = 432000  # 持有时间
    sma_rolling_window = 20  # 简单移动平均窗口
    std_rolling_window = 40  # 标准差窗口
    qty = 0.002  # 交易量
    entry_time = datetime.now(tz=timezone.utc).timestamp()  # 进入时间

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    def __init__(self):
        self.position = {"long": {"quantity": 0.0}}

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "holding_time":
            self.holding_time = int(value)
        elif identifier == "sma_rolling_window":
            self.sma_rolling_window = int(value)
        elif identifier == "std_rolling_window":
            self.std_rolling_window = int(value)
        else:
            logging.error(f"Could not set {identifier}, not found")

    def get_mean(self, array):
        return np.mean(array)

    def get_stddev(self, array):
        return np.std(array, ddof=1)

    def get_rolling_mean(self, array, rolling_window):
        return np.concatenate(([0] * (rolling_window - 1), np.nan_to_num(pd.Series(array).rolling(window=rolling_window).mean().to_numpy())))

    def get_rolling_std(self, array, rolling_window):
        return np.concatenate(([0] * (rolling_window - 1), np.nan_to_num(pd.Series(array).rolling(window=rolling_window).std(ddof=1).to_numpy())))

    def convert_ms_to_datetime(self, milliseconds):
        seconds = milliseconds / 1000.0
        return datetime.fromtimestamp(seconds, tz=timezone.utc)

    async def on_datasource_interval(self, netflow_data):
        netflow_start_time = np.array([float(c["end_time"]) for c in netflow_data])
        netflow_total = np.array([float(c["netflow_total"]) for c in netflow_data])
        sma_netflow = self.get_rolling_mean(netflow_total, self.sma_rolling_window)
        std_sma_netflow = self.get_rolling_std(sma_netflow, self.std_rolling_window)

        logging.info(f"current_position: {self.position}, netflow_total: {netflow_total[-1]}, sma_netflow: {sma_netflow[-1]}, std_sma_netflow: {std_sma_netflow[-1]}, first data : {netflow_data[0]} at {self.convert_ms_to_datetime(netflow_start_time[0])}, last data : {netflow_data[-1]} at {self.convert_ms_to_datetime(netflow_start_time[-1])}, length: {len(netflow_total)}")

        if self.position["long"]["quantity"] > 0.0 and netflow_start_time[-1] - self.entry_time >= self.holding_time * 1000:
            await self.close_position(netflow_start_time[-1], netflow_total, sma_netflow, std_sma_netflow)

        if sma_netflow[-1] < std_sma_netflow[-1] and self.position["long"]["quantity"] <= 0.0:
            await self.open_position(netflow_start_time[-1], netflow_total, sma_netflow, std_sma_netflow)

    async def close_position(self, timestamp, netflow_total, sma_netflow, std_sma_netflow):
        self.position["long"]["quantity"] -= self.qty
        logging.info(f"Closed long position with qty {self.qty} when netflow_total: {netflow_total[-1]}, sma_netflow: {sma_netflow[-1]}, std_sma_netflow: {std_sma_netflow[-1]} at time {self.convert_ms_to_datetime(timestamp)}")

    async def open_position(self, timestamp, netflow_total, sma_netflow, std_sma_netflow):
        self.entry_time = timestamp
        self.position["long"]["quantity"] += self.qty
        logging.info(f"Placed a buy order with qty {self.qty} when netflow_total: {netflow_total[-1]}, sma_netflow: {sma_netflow[-1]}, std_sma_netflow: {std_sma_netflow[-1]} at time {self.convert_ms_to_datetime(timestamp)}")

# 模拟数据
netflow_data = [
    {"end_time": 1609459200000, "netflow_total": 100},
    {"end_time": 1609545600000, "netflow_total": 150},
    # 添加更多模拟数据
]

strategy = Strategy()

# 模拟运行
asyncio.run(strategy.on_datasource_interval(netflow_data))
