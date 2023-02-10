import numpy as np
import statistics
import pandas as pd
from datetime import datetime, timedelta
import time
import ta
import matplotlib.pyplot as plt
from pybit import usdt_perpetual


class Technical_Indicators():
    def __init__(self):
        print("Technical Indicators Loaded")

    def MACD(self,df):
        macd = ta.trend.MACD(df.Close).macd_diff()
        df['MACD'] = macd.values
        return df

    def StochRSI(self,df, period=14, smoothK=3, smoothD=3):
    # Calculate RSI 
        delta = df.Close.diff().dropna()
        ups = delta * 0
        downs = ups.copy()
        ups[delta > 0] = delta[delta > 0]
        downs[delta < 0] = -delta[delta < 0]
        ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
        ups = ups.drop(ups.index[:(period-1)])
        downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
        downs = downs.drop(downs.index[:(period-1)])
        rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
            downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() 
        rsi = 100 - 100 / (1 + rs)

        # Calculate StochRSI 
        stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        stochrsi_K = stochrsi.rolling(smoothK).mean()
        stochrsi_D = stochrsi_K.rolling(smoothD).mean()

        df['SRSI'] = stochrsi
        df['RSI_K'] = stochrsi_D
        df['RSI_D'] = stochrsi_K
        return df
    
    def reduceDiff(self,df,_loc):
        diff = np.insert(np.diff(df[_loc]), 0, 0)
        df[_loc] = diff
        return df

    def closeDiff(self,df):
        diff = np.insert(np.diff(df['Close']), 0, 0)
        df['Diff'] = diff
        return df
        
    def SMAmedians(self,df):
        dfList = df.columns.to_list()
        for column in dfList:
            if "SMA_" in column:
                ma_med = []
                for i in range(0,len(df)):
                    ma_med.append(float(df['SMA_5'][i]-df['Close'][i]))
                df['median_' + column] = ma_med
                df['median_' + column] = df['median_' + column].pct_change()
        return df

    def BollingerBands(self,df):
        sma = df.Close.rolling(window=5).mean().dropna()
        rstd = df.Close.rolling(window=5).std().dropna()
        upper_band = sma + 2 * rstd
        lower_band = sma - 2 * rstd
        df['Boll_up'] = upper_band
        df['Boll_down'] = lower_band
        bollmed = []
        for i in range(0,len(df)):
            bollmed.append(float(statistics.median([df['Boll_down'][i],df['Boll_up'][i]])))
        df['Boll_Med'] = bollmed
        return df

    def MovingAverage(self,df,ranges):
        for i in ranges:
            df['SMA_' + str(i)] = df.Close.rolling(window=i).mean()
        return df

    def VolumeMedian(self,df,ranges):
        for i in ranges:
            df['volume_median_' + str(i)] = df.volume.rolling(window=i).mean().pct_change()
        return df

    def nearest_number(self,num, list_numbers):
        return min(list_numbers, key=lambda x: abs(x-num))

    def nearest_larger_number(self,num, num_list):
        import math
        closest = None
        min_diff = float("inf")
        for n in num_list:
            diff = n - num
            if diff > 0 and diff < min_diff:
                closest = n
                min_diff = diff
        return closest

    def _rangeReduced(self,df,pivots,diff,upper=False,lower=False):
        pivots_down_values = [x[0] for x in pivots]
        data_band = []
        Values_table = []
        for pos,row in df.iterrows():
            result = [x for x in pivots_down_values if x < pos] #obtenemos las resistencias anteriores
            result = [x for x in result if (result[-1] - x) < 1000] #nos quedamos solo con los valores de los ultimos 1000 datos
            if len(result) > 0:
                L_row = []
                for r in result:
                    L_row.append([x for x in pivots if r == x[0]][0][1]) #obtenemos el valor de la resitencia y agregamos a L_row
                if upper:near = self.nearest_larger_number(row['Close'],L_row) #calculamos el valor mas cercano entre Close y los valores de L_row
                if lower:near = self.nearest_number(row['Close'],L_row) 
                if near is not None:data_band.append(near)
                else:data_band.append(data_band[-1])
            else:data_band.append(pivots[0][1])
        if not diff:
            for i in range(len(df)):
                    Values_table.append(data_band[i] - list(df['Close'].values)[i])
            return Values_table
        else:return data_band

    def __is_far_from_level(self,value, levels, df):    
        ave =  np.mean(df['high'] - df['low'])    
        return np.sum([abs(value-level)<ave for _,level in levels])==0

    def __getPivotsUp(self,df,_range,limit):
        pivots = []
        max_list = []
        if not limit:limit = len(df)
        for i in range((len(df)-limit)+_range, len(df)-_range):
            high_range = df['Close'][i-_range:i+(_range-1)]
            current_max = high_range.max()
            if current_max not in max_list:
                max_list = []
            max_list.append(current_max)
            if len(max_list)==_range and self.__is_far_from_level(current_max,pivots,df):
                pivots.append((high_range.idxmax(), current_max))
        return pivots
    
    def __getPivotsDown(self,df,_range,limit):
        pivots = []
        min_list = []
        if not limit:limit = len(df)
        for i in range((len(df)-limit)+_range, len(df)-_range):            
            low_range = df['Close'][i-_range:i+_range]
            current_min = low_range.min()
            if current_min not in min_list:
                min_list = []
            min_list.append(current_min)
            if len(min_list)==5 and self.__is_far_from_level(current_min,pivots,df):
                pivots.append((low_range.idxmin(), current_min))
        return pivots

    def resistanceAndSupports(self,df,_range=100,limit=False,diff=False,p_diff=False):
        resistances = self.__getPivotsUp(df,_range,limit)
        supports = self.__getPivotsDown(df,_range,limit)
        if diff:
            resistances = self._rangeReduced(df,resistances,p_diff,upper=True)
            supports = self._rangeReduced(df,supports,p_diff,lower=True)
        df['support'] = supports
        # df['support'] = df['support'].pct_change()
        df['resistances'] = resistances
        # df['resistances'] = df['resistances'].pct_change()
        return df


    def reduceToClose(self,df,_loc):
        values_table = []
        for i in range(len(df)):
            values_table.append(list(df[_loc].values)[i] - list(df['Close'].values)[i])
        df[_loc] = values_table
        return df

################

class Utils():
    def __init__(self):
        print("Utils Module loaded")

    def downloadData(self,_coin,session):
        _datasheet = {}
        dataframe = []
        high = []
        low = []
        volume = []
        timestamp = []
        for i in range(2000,0,-50):    
            dt = datetime.today() - timedelta(hours=i)
            startTime = str(round(time.mktime(dt.timetuple())))
            klines = session.query_kline(symbol=_coin,interval='15',from_time=startTime,limit=200)
            for res in klines['result']:
                dataframe.append(float(res['close']))
                high.append(float(res['high']))
                low.append(float(res['low']))
                volume.append(float(res['volume']))
                # timestamp.append(res['open_time'])s
        return pd.DataFrame({'Close':dataframe,'high':high,'low':low,'volume':volume})# 'time':timestamp})

    def getSession(self):
        api_key='bw4DqRtbexQ2ClDX54'
        api_secret='AmGaN3QLhimm45lF490v4WKb2Gbw1mkJsUCg'
        _endpoint = "https://api.bybit.com"
        session = usdt_perpetual.HTTP(endpoint=_endpoint, api_key=api_key, api_secret=api_secret)
        unauth_session = usdt_perpetual.HTTP(endpoint=_endpoint, recv_window=5000)
        return session,unauth_session



################

class Drawing():
    def __init__(self):
        print("Drawing Module loaded")

    def plot(self,df,pivots_up=False,pivots_down=False,_name = 'COIN'):
        #plt.figure(2)
        fig, axs = plt.subplots(2,gridspec_kw={'height_ratios': [10,3]})
        
        axs[0].plot(df.index, df['Close'], label='Close', color='blue')
        axs[0].plot(df.index, df['pivots_up'], label='Close', color='green')
        axs[0].plot(df.index, df['pivots_down'], label='Close', color='red')
        # axs[0].scatter(df.index, df['buy'], color='green', label='Buy', marker='^', alpha = 1)
        # axs[0].scatter(df.index, df['sell'], color='red', label='Sell', marker='v', alpha = 1)
        axs[0].set_title('Price History based on Close of ' + _name)
        axs[0].set(xlabel = 'Date', ylabel = 'Price USD')

        #plt.figure(1)
        # axs[1].plot(df.index, df['RSI'],label="RSI(14)", color='orange')
        if pivots_up:
            for pivot in pivots_up:
                axs[0].axhline(pivot[1], linestyle='--', alpha=0.5,color='blue')
        if pivots_down:
            for pivot in pivots_down:
                axs[0].axhline(pivot[1], linestyle='--', alpha=0.5,color='red')
        # axs[1].axhline(50, linestyle='--', alpha=0.5)
        # axs[1].axhline(30, linestyle='--', alpha=0.5)
        # axs[1].set_title('RSI History')
        # axs[1].legend(loc='upper left')
        # axs[1].set(xlabel = 'Date', ylabel = 'RSI Value')


        # axs[1].plot(df.index, df['prediction'],label="Median", color='green')
        # axs[1].plot(df.index, df['pct_n_low'],label="pct", color='green')
        # axs[1].plot(df.index, df['median_MA'],label="pct", color='red')
        # axs[1].plot(df.index, df['dayly_SMA_hi_pct'],label="SMA(99)", color='green')
        # axs[1].set_ylim([-0.1,0.1])
        # axs[2].plot(df.index, df['hourly_SMA_cal'], label='Close', color='blue')
        plt.show();