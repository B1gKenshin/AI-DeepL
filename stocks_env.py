import numpy as np

from .trading_env import TradingEnv, Actions, Positions

from sklearn.preprocessing import StandardScaler




class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, new_frame = 0):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.new_frame = new_frame
        self.new_frame_cont = self.new_frame
        super().__init__(df, window_size)
        

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def __getRange(self,prices):
        origin = self.new_frame_cont
        dest = self.new_frame_cont + self.frame_bound[1]
        if len(prices[origin:dest]) < self.frame_bound[1]:
            self.new_frame_cont = self.new_frame
            origin = self.new_frame_cont
            dest = self.new_frame_cont + self.frame_bound[1]
        return origin,dest

    def _prepare_data(self,df):
        MAranges = [5,10,25,50,200]
        df = self.ti.MovingAverage(df,MAranges)
        df = self.ti.closeDiff(df)
        # df = self.ti.SMAmedians(df)
        df = self.ti.BollingerBands(df)
        df = self.ti.StochRSI(df)
        df = self.ti.MACD(df)
        # df = self.ti.VolumeMedian(df,MAranges)
        # df = self.ti.resistanceAndSupports(df,diff=True)
        df = self.ti.reduceToClose(df,'low')
        df = self.ti.reduceToClose(df,'high')
        # df = self.ti.reduceDiff(df,'volume')
        df = df.iloc[MAranges[-1]:len(df)]
        df = df.reset_index()
        return df

    def _process_data(self):
        prices = self.df['Close'].to_numpy()
        origin,dest = self.__getRange(prices)
        prices = prices[origin:dest]
        # signal_features = self.df.loc[ : , self.df.columns != 'Close']
        signal_features = self.df.loc[ : , self.df.columns != 'Unnamed: 0.1']
        signal_features = signal_features.loc[ : , signal_features.columns != 'Unnamed: 0']
        signal_features = signal_features.loc[ : , signal_features.columns != 'time']
        signal_features = signal_features.iloc[origin:dest]
        sc_X = StandardScaler()
        signal_features = sc_X.fit_transform(signal_features)
        self.new_frame_cont += self.new_frame
        return prices, signal_features


    def _calculate_reward(self):
        step_reward = 0

        if self._position == Positions.Long:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price
            price_diff_percent = 100 * (price_diff / last_trade_price)
            holding_period = self._current_tick - self._last_trade_tick

            if price_diff > 0: 
                step_reward += price_diff_percent / holding_period
            else:
                step_reward -= price_diff_percent / holding_period
        # self._position == Positions.Nothing
        return step_reward

    def _update_profit(self):
        self._total_profit = self._total_profit+self._total_reward

