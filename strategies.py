import pandas as pd
import pandas_ta as ta
import numpy as np

class StrategyBase:
    """
    Base class for all trading strategies.
    Subclasses must implement the generate_signals method.
    """
    def __init__(self, data):
        # The backtester will pass historical data to the strategy's __init__
        self.data = data.copy()

    def generate_signals(self):
        """
        Generates trading signals (1 for Buy, -1 for Sell, 0 for Hold)
        based on the strategy's logic and the provided historical self.data.

        Parameters:
        - self.data (pd.DataFrame): The historical market self.data (Date, Open, High, Low, Close, Adj Close, Volume).

        Returns:
        - pd.Series: A Series with the same index as 'self.data', containing
                     1 (Buy), -1 (Sell), or 0 (Hold) signals.
                     Ensure the Series contains only integer values.
        """
        raise NotImplementedError("Subclasses must implement generate_signals() method.")


class ADXWilliamsStrategy(BaseStrategy):
    """
    Implements the ADX Williams trading strategy.
    
    This strategy combines the Directional Movement Index (ADX) for trend strength,
    DI+ and DI- for trend direction, and the Williams %R oscillator for overbought/oversold
    conditions. A Simple Moving Average (SMA) is used as a stop-loss mechanism.
    
    Timeframe Adaptation:
    - Originally a scalping strategy, it has been adapted for a daily timeframe.
    - Indicator periods (ADX: 14, Williams %R: 14, SMA: 50) are standard and suitable for daily charts.
    - The take-profit rule (30-period high/low) is also applied to the daily timeframe.
    """
    def generate_signals(self):
        """
        Generates trading signals based on the ADX Williams strategy rules.
        
        Returns:
            pd.Series: A Series containing trading signals (1 for buy, -1 for sell, 0 for hold).
        """
        # --- 1. Calculate Indicators ---
        adx_indicator = self.data.ta.adx(length=14)
        self.data['ADX_14'] = adx_indicator['ADX_14']
        self.data['DMP_14'] = adx_indicator['DMP_14']
        self.data['DMN_14'] = adx_indicator['DMN_14']
        self.data['WILLR_14'] = self.data.ta.willr(length=14)
        self.data['SMA_50'] = self.data.ta.sma(length=50)
        self.data['High_30'] = self.data['High'].rolling(window=30).max()
        self.data['Low_30'] = self.data['Low'].rolling(window=30).min()

        # --- 2. Generate Signals (Stateful Loop) ---
        signals = pd.Series(0, index=self.data.index)
        position = 0  # 0: no position, 1: long, -1: short

        for i in range(1, len(self.data)):
            is_uptrend = self.data['DMP_14'].iloc[i] > self.data['DMN_14'].iloc[i]
            is_downtrend = self.data['DMN_14'].iloc[i] > self.data['DMP_14'].iloc[i]
            is_strong_trend = self.data['ADX_14'].iloc[i] > 50
            is_oversold = self.data['WILLR_14'].iloc[i] < -80
            is_overbought = self.data['WILLR_14'].iloc[i] > -20
            
            long_entry_condition = is_uptrend and is_strong_trend and is_oversold
            short_entry_condition = is_downtrend and is_strong_trend and is_overbought

            if position == 1:
                if self.data['Close'].iloc[i] < self.data['SMA_50'].iloc[i] or \
                   self.data['Close'].iloc[i] >= self.data['High_30'].iloc[i-1] or \
                   short_entry_condition:
                    signals.iloc[i] = -1
                    position = -1 if short_entry_condition else 0
            elif position == -1:
                if self.data['Close'].iloc[i] > self.data['SMA_50'].iloc[i] or \
                   self.data['Close'].iloc[i] <= self.data['Low_30'].iloc[i-1] or \
                   long_entry_condition:
                    signals.iloc[i] = 1
                    position = 1 if long_entry_condition else 0
            
            if position == 0:
                if long_entry_condition:
                    signals.iloc[i] = 1
                    position = 1
                elif short_entry_condition:
                    signals.iloc[i] = -1
                    position = -1
        
        self.signals = signals
        return self.signals

class TrendWaveRiderStrategy(BaseStrategy):
    """
    Implements the Trend Wave Rider strategy.

    Confirms a trend using three distinct methods: ADX, a crossover of two SMAs (25/50),
    and the CCI as an entry trigger.
    
    Timeframe Adaptation:
    - Adapted from shorter timeframes to daily data using standard indicator lengths.
    """
    def generate_signals(self):
        """Generates trading signals."""
        adx_indicator = self.data.ta.adx(length=14)
        self.data['ADX_14'] = adx_indicator['ADX_14']
        self.data['DMP_14'] = adx_indicator['DMP_14']
        self.data['DMN_14'] = adx_indicator['DMN_14']
        self.data['SMA_25'] = self.data.ta.sma(length=25)
        self.data['SMA_50'] = self.data.ta.sma(length=50)
        self.data['CCI_20'] = self.data.ta.cci(length=20)
        
        adx_trend_up = (self.data['DMP_14'] > self.data['DMN_14']) & (self.data['ADX_14'] > 50)
        adx_trend_down = (self.data['DMN_14'] > self.data['DMP_14']) & (self.data['ADX_14'] > 50)
        ma_trend_up = self.data['SMA_25'] > self.data['SMA_50']
        ma_trend_down = self.data['SMA_25'] < self.data['SMA_50']
        cci_oversold = self.data['CCI_20'] < -100
        cci_overbought = self.data['CCI_20'] > 100
        
        long_entry = adx_trend_up & ma_trend_up & cci_oversold
        short_entry = adx_trend_down & ma_trend_down & cci_overbought
        
        signals = pd.Series(0, index=self.data.index)
        signals[long_entry] = 1
        signals[short_entry] = -1
        
        position = 0
        for i in range(len(signals)):
            if signals.iloc[i] == 1:
                if position == 1: signals.iloc[i] = 0
                else: position = 1
            elif signals.iloc[i] == -1:
                if position == -1: signals.iloc[i] = 0
                else: position = -1
        
        self.signals = signals
        return self.signals

class CloudScalperStrategy(BaseStrategy):
    """
    Implements the Cloud Scalper strategy.

    Uses Ichimoku Cloud, filtered by ADX, Bollinger Band Width (BBW), and a long-term EMA.
    
    Timeframe Adaptation:
    - "Anchor timeframe" 200-period EMA adapted to the daily chart.
    """
    def generate_signals(self):
        """Generates trading signals."""
        ichimoku = self.data.ta.ichimoku(tenkan=9, kijun=26, senkou=52)
        self.data['TENKAN'] = ichimoku[0]
        self.data['KIJUN'] = ichimoku[1]
        self.data['SPAN_A'] = ichimoku[2]
        self.data['SPAN_B'] = ichimoku[3]
        self.data['ADX_14'] = self.data.ta.adx(length=14)['ADX_14']
        bbands = self.data.ta.bbands(length=20)
        self.data['BBW'] = bbands['BBB_20_2.0']
        self.data['EMA_200'] = self.data.ta.ema(length=200)

        tenkan_above_kijun = self.data['TENKAN'] > self.data['KIJUN']
        span_a_above_span_b = self.data['SPAN_A'] > self.data['SPAN_B']
        strong_trend = self.data['ADX_14'] > 50
        volatility_not_extreme = self.data['BBW'] < 5.0
        long_term_uptrend = self.data['Close'] > self.data['EMA_200']
        
        long_entry = tenkan_above_kijun & span_a_above_span_b & strong_trend & volatility_not_extreme & long_term_uptrend
        short_entry = (~tenkan_above_kijun) & (~span_a_above_span_b) & strong_trend & volatility_not_extreme & (~long_term_uptrend)

        signals = pd.Series(0, index=self.data.index)
        signals[long_entry] = 1
        signals[short_entry] = -1
        
        position = 0
        for i in range(len(signals)):
            if signals.iloc[i] == 1:
                if position == 1: signals.iloc[i] = 0
                else: position = 1
            elif signals.iloc[i] == -1:
                if position == -1: signals.iloc[i] = 0
                else: position = -1
        
        self.signals = signals
        return self.signals

class TurtleStrategy(BaseStrategy):
    """
    Implements a modified Turtle Trading strategy.

    Uses a Donchian Channel breakout, filtered by a long-term SMA, ADX, and Chop Index.
    
    Timeframe Adaptation:
    - Core 20-day breakout logic is naturally suited for daily charts.
    """
    def generate_signals(self):
        """Generates trading signals."""
        donchian = self.data.ta.donchian(lower_length=20, upper_length=20)
        self.data['UPPER_20_PREV'] = donchian['DCU_20_20'].shift(1)
        self.data['LOWER_20_PREV'] = donchian['DCL_20_20'].shift(1)
        self.data['SMA_200'] = self.data.ta.sma(length=200)
        self.data['ADX_14'] = self.data.ta.adx(length=14)['ADX_14']
        self.data['CHOP_14'] = self.data.ta.chop(length=14)

        long_breakout = self.data['Close'] > self.data['UPPER_20_PREV']
        short_breakout = self.data['Close'] < self.data['LOWER_20_PREV']
        long_term_uptrend = self.data['Close'] > self.data['SMA_200']
        trending_market_adx = self.data['ADX_14'] > 30
        not_choppy = self.data['CHOP_14'] < 40
        
        long_entry = long_breakout & long_term_uptrend & trending_market_adx & not_choppy
        short_entry = short_breakout & (~long_term_uptrend) & trending_market_adx & not_choppy

        signals = pd.Series(0, index=self.data.index)
        signals[long_entry] = 1
        signals[short_entry] = -1
        
        position = 0
        for i in range(len(signals)):
            if signals.iloc[i] == 1:
                if position == 1: signals.iloc[i] = 0
                else: position = 1
            elif signals.iloc[i] == -1:
                if position == -1: signals.iloc[i] = 0
                else: position = -1

        self.signals = signals
        return self.signals

# --- Newly Added 6 Strategies ---

class KAMAStrategy(BaseStrategy):
    """
    Implements a strategy based on Kaufman's Adaptive Moving Average (KAMA).

    This trend-following strategy uses KAMA for primary trend direction, confirmed by a longer-term KAMA.
    It is filtered by ADX, Chop Index, and Bollinger Band Width to improve signal quality.
    A cooldown period after a trade is closed prevents immediate re-entry.
    
    Timeframe Adaptation:
    - The "bigger timeframe" KAMA (from a 4h chart) is adapted to a long-period daily KAMA (100).
    - ATR-based SL/TP is adapted to a signal reversal exit model for compatibility.
    """
    def generate_signals(self):
        """Generates trading signals."""
        # --- 1. Calculate Indicators ---
        self.data['KAMA_14'] = self.data.ta.kama(length=14)
        self.data['KAMA_100'] = self.data.ta.kama(length=100) # Long-term KAMA
        self.data['ADX_50'] = self.data.ta.adx(length=50)['ADX_50']
        self.data['CHOP_14'] = self.data.ta.chop(length=14)
        bbands = self.data.ta.bbands(length=20)
        self.data['BBW_20'] = bbands['BBB_20_2.0']

        # --- 2. Define Conditions ---
        price_above_kama = self.data['Close'] > self.data['KAMA_14']
        kama_uptrend = self.data['KAMA_14'] > self.data['KAMA_100']
        strong_trend_adx = self.data['ADX_50'] > 50
        not_choppy = self.data['CHOP_14'] < 50
        volatility_filter = self.data['BBW_20'] < 7.0

        long_entry = price_above_kama & kama_uptrend & strong_trend_adx & not_choppy & volatility_filter
        short_entry = (~price_above_kama) & (~kama_uptrend) & strong_trend_adx & not_choppy & volatility_filter
        
        # --- 3. Generate Signals with Cooldown ---
        signals = pd.Series(0, index=self.data.index)
        position = 0
        last_trade_candle = -10 # Initialize to allow immediate first trade

        for i in range(len(self.data)):
            cooldown_period = 10
            can_trade = (i - last_trade_candle) >= cooldown_period

            if position == 0 and can_trade:
                if long_entry.iloc[i]:
                    signals.iloc[i] = 1
                    position = 1
                elif short_entry.iloc[i]:
                    signals.iloc[i] = -1
                    position = -1
            elif position == 1 and not long_entry.iloc[i]: # Exit condition
                signals.iloc[i] = -1
                position = 0
                last_trade_candle = i
            elif position == -1 and not short_entry.iloc[i]: # Exit condition
                signals.iloc[i] = 1
                position = 0
                last_trade_candle = i

        self.signals = signals
        return self.signals

class AlligatorStrategy(BaseStrategy):
    """
    Implements the Alligator trading strategy.

    Uses the Williams Alligator indicator for trend direction. Signals are filtered
    by ADX, a long-term EMA, Chande Momentum Oscillator (CMO), and Stochastic RSI.
    
    Timeframe Adaptation:
    - The higher timeframe (4h) EMA is adapted to a long-period daily EMA (100).
    - Indicator settings are standard for daily charts.
    """
    def generate_signals(self):
        """Generates trading signals."""
        # --- 1. Calculate Indicators ---
        alligator = self.data.ta.alligator()
        self.data['JAW'] = alligator['jaw_13_8']
        self.data['TEETH'] = alligator['teeth_8_5']
        self.data['LIPS'] = alligator['lips_5_3']
        self.data['ADX_14'] = self.data.ta.adx(length=14)['ADX_14']
        self.data['EMA_100'] = self.data.ta.ema(length=100)
        self.data['CMO_14'] = self.data.ta.cmo(length=14)
        stoch_rsi = self.data.ta.stochrsi(length=14)
        self.data['STOCHRSI_K'] = stoch_rsi['STOCHRSIk_14_14_3_3']
        
        # --- 2. Define Conditions ---
        alligator_open_up = (self.data['LIPS'] > self.data['TEETH']) & (self.data['TEETH'] > self.data['JAW'])
        price_above_alligator = self.data['Close'] > self.data['LIPS']
        
        alligator_open_down = (self.data['LIPS'] < self.data['TEETH']) & (self.data['TEETH'] < self.data['JAW'])
        price_below_alligator = self.data['Close'] < self.data['LIPS']
        
        trending_adx = self.data['ADX_14'] > 30
        long_term_up = self.data['Close'] > self.data['EMA_100']
        cmo_up = self.data['CMO_14'] > 20
        srsi_oversold = self.data['STOCHRSI_K'] < 20
        
        cmo_down = self.data['CMO_14'] < -20
        srsi_overbought = self.data['STOCHRSI_K'] > 80

        long_entry = alligator_open_up & price_above_alligator & trending_adx & long_term_up & cmo_up & srsi_oversold
        short_entry = alligator_open_down & price_below_alligator & trending_adx & ~long_term_up & cmo_down & srsi_overbought
        
        # --- 3. Generate Signals ---
        signals = pd.Series(0, index=self.data.index)
        signals[long_entry] = 1
        signals[short_entry] = -1

        position = 0
        for i in range(len(signals)):
            if signals.iloc[i] == 1:
                if position == 1: signals.iloc[i] = 0
                else: position = 1
            elif signals.iloc[i] == -1:
                if position == -1: signals.iloc[i] = 0
                else: position = -1
        
        self.signals = signals
        return self.signals

class IchimokuCloudStrategy(BaseStrategy):
    """
    Implements the Ichimoku Cloud strategy from the transcript.

    This is a trend-following strategy using the Ichimoku Cloud's components
    (Tenkan/Kijun cross and the Cloud itself). It's filtered by ADX and Chop Index.
    
    Timeframe Adaptation:
    - All indicator settings are standard and well-suited for daily analysis.
    - Exit logic is adapted to a signal reversal model.
    """
    def generate_signals(self):
        """Generates trading signals."""
        # --- 1. Calculate Indicators ---
        ichimoku = self.data.ta.ichimoku(tenkan=9, kijun=26, senkou=52)
        self.data['TENKAN'] = ichimoku[0]
        self.data['KIJUN'] = ichimoku[1]
        self.data['SPAN_A'] = ichimoku[2]
        self.data['SPAN_B'] = ichimoku[3]
        self.data['ADX_14'] = self.data.ta.adx(length=14)['ADX_14']
        self.data['CHOP_14'] = self.data.ta.chop(length=14)

        # --- 2. Define Conditions ---
        short_term_bullish = self.data['TENKAN'] > self.data['KIJUN']
        long_term_bullish = self.data['SPAN_A'] > self.data['SPAN_B']
        trending_market = self.data['ADX_14'] > 50
        not_choppy = self.data['CHOP_14'] < 50

        long_entry = short_term_bullish & long_term_bullish & trending_market & not_choppy
        short_entry = (~short_term_bullish) & (~long_term_bullish) & trending_market & not_choppy

        # --- 3. Generate Signals ---
        signals = pd.Series(0, index=self.data.index)
        signals[long_entry] = 1
        signals[short_entry] = -1
        
        position = 0
        for i in range(len(signals)):
            if signals.iloc[i] == 1:
                if position == 1: signals.iloc[i] = 0
                else: position = 1
            elif signals.iloc[i] == -1:
                if position == -1: signals.iloc[i] = 0
                else: position = -1
        
        self.signals = signals
        return self.signals

class TrendFollowingAIStrategy(BaseStrategy):
    """
    Implements the trend-following strategy described as being generated by an AI.

    It uses a dual EMA crossover, MACD, and ADX for signal generation,
    confirmed by the same conditions on a longer timeframe.
    
    Timeframe Adaptation:
    - The "big trend" analysis from a 4h chart is adapted using longer-period daily EMAs (50/100).
    """
    def generate_signals(self):
        """Generates trading signals."""
        # --- 1. Calculate Indicators ---
        # Short-term trend
        self.data['EMA_10'] = self.data.ta.ema(length=10)
        self.data['EMA_20'] = self.data.ta.ema(length=20)
        macd = self.data.ta.macd(fast=12, slow=26, signal=9)
        self.data['MACD_HIST'] = macd['MACDh_12_26_9']
        self.data['ADX_14'] = self.data.ta.adx(length=14)['ADX_14']
        
        # Long-term trend confirmation
        self.data['EMA_50'] = self.data.ta.ema(length=50)
        self.data['EMA_100'] = self.data.ta.ema(length=100)

        # --- 2. Define Conditions ---
        short_term_up = (self.data['EMA_10'] > self.data['EMA_20']) & (self.data['MACD_HIST'] > 0)
        long_term_up = self.data['EMA_50'] > self.data['EMA_100']
        trending_market = self.data['ADX_14'] > 40
        
        long_entry = short_term_up & long_term_up & trending_market
        short_entry = (~short_term_up) & (~long_term_up) & trending_market

        # --- 3. Generate Signals ---
        signals = pd.Series(0, index=self.data.index)
        signals[long_entry] = 1
        signals[short_entry] = -1

        position = 0
        for i in range(len(signals)):
            if signals.iloc[i] == 1:
                if position == 1: signals.iloc[i] = 0
                else: position = 1
            elif signals.iloc[i] == -1:
                if position == -1: signals.iloc[i] = 0
                else: position = -1
        
        self.signals = signals
        return self.signals

class SwingEMAStrategy(BaseStrategy):
    """
    Implements a swing trading strategy using a triple EMA crossover.

    The trend is defined by the alignment of three EMAs (21, 50, 100). The ADX
    is used as a filter to only trade in trending markets.
    
    Timeframe Adaptation:
    - The EMA periods are well-suited for daily swing trading.
    - The complex partial take-profit/trailing stop exit is simplified to a
      signal reversal model to fit the backtester.
    """
    def generate_signals(self):
        """Generates trading signals."""
        # --- 1. Calculate Indicators ---
        self.data['EMA_21'] = self.data.ta.ema(length=21)
        self.data['EMA_50'] = self.data.ta.ema(length=50)
        self.data['EMA_100'] = self.data.ta.ema(length=100)
        self.data['ADX_14'] = self.data.ta.adx(length=14)['ADX_14']
        
        # --- 2. Define Conditions ---
        ema_aligned_up = (self.data['EMA_21'] > self.data['EMA_50']) & (self.data['EMA_50'] > self.data['EMA_100'])
        price_confirm_up = self.data['Close'] > self.data['EMA_21']
        
        ema_aligned_down = (self.data['EMA_21'] < self.data['EMA_50']) & (self.data['EMA_50'] < self.data['EMA_100'])
        price_confirm_down = self.data['Close'] < self.data['EMA_21']
        
        trending_market = self.data['ADX_14'] > 25

        long_entry = ema_aligned_up & price_confirm_up & trending_market
        short_entry = ema_aligned_down & price_confirm_down & trending_market

        # --- 3. Generate Signals ---
        signals = pd.Series(0, index=self.data.index)
        signals[long_entry] = 1
        signals[short_entry] = -1

        position = 0
        for i in range(len(signals)):
            if signals.iloc[i] == 1:
                if position == 1: signals.iloc[i] = 0
                else: position = 1
            elif signals.iloc[i] == -1:
                if position == -1: signals.iloc[i] = 0
                else: position = -1
        
        self.signals = signals
        return self.signals

class SupertrendScalperStrategy(BaseStrategy):
    """
    Implements a scalping strategy using the Supertrend indicator.

    Uses a primary Supertrend for entry signals, confirmed by a longer-term Supertrend
    and a long-term EMA. ADX filters for trend strength.
    
    Timeframe Adaptation:
    - Adapted from a 15-minute to a daily timeframe.
    - Long-term indicators (from a 4h chart) are adapted using a Supertrend with a larger
      multiplier and a 200-day EMA.
    - ATR-based SL/TP is simplified to a signal reversal exit.
    """
    def generate_signals(self):
        """Generates trading signals."""
        # --- 1. Calculate Indicators ---
        supertrend_short = self.data.ta.supertrend(length=7, multiplier=3)
        self.data['SUPERT_7_3'] = supertrend_short['SUPERT_7_3.0']
        
        supertrend_long = self.data.ta.supertrend(length=14, multiplier=5) # Longer-term supertrend
        self.data['SUPERT_14_5'] = supertrend_long['SUPERT_14_5.0']
        
        self.data['EMA_200'] = self.data.ta.ema(length=200)
        self.data['ADX_14'] = self.data.ta.adx(length=14)['ADX_14']

        # --- 2. Define Conditions ---
        short_term_up = self.data['Close'] > self.data['SUPERT_7_3']
        long_term_up_supertrend = self.data['Close'] > self.data['SUPERT_14_5']
        long_term_up_ema = self.data['Close'] > self.data['EMA_200']
        trending_market = self.data['ADX_14'] > 30

        long_entry = short_term_up & long_term_up_supertrend & long_term_up_ema & trending_market
        short_entry = (~short_term_up) & (~long_term_up_supertrend) & (~long_term_up_ema) & trending_market

        # --- 3. Generate Signals ---
        signals = pd.Series(0, index=self.data.index)
        signals[long_entry] = 1
        signals[short_entry] = -1

        position = 0
        for i in range(len(signals)):
            if signals.iloc[i] == 1:
                if position == 1: signals.iloc[i] = 0
                else: position = 1
            elif signals.iloc[i] == -1:
                if position == -1: signals.iloc[i] = 0
                else: position = -1
        
        self.signals = signals
        return self.signals


# -----------------------------------------------------------------------------
# Strategy 1: Golden Cross / Death Cross Strategy
# -----------------------------------------------------------------------------
class Strategy1_GoldenCrossStrategy(StrategyBase):
    """
    Strategy based on the crossover of two Simple Moving Averages (SMAs).
    - A "Golden Cross" (Buy signal) occurs when the short-term SMA crosses above the long-term SMA.
    - A "Death Cross" (Sell signal) occurs when the short-term SMA crosses below the long-term SMA.
    """
    def __init__(self, data, short_window=50, long_window=200):
        super().__init__(data)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0

        # Calculate short and long term Simple Moving Averages
        self.data['SMA_short'] = ta.sma(self.data['Close'], length=self.short_window)
        self.data['SMA_long'] = ta.sma(self.data['Close'], length=self.long_window)

        # Generate buy signals
        signals['buy'] = (self.data['SMA_short'] > self.data['SMA_long']) & (self.data['SMA_short'].shift(1) <= self.data['SMA_long'].shift(1))
        # Generate sell signals
        signals['sell'] = (self.data['SMA_short'] < self.data['SMA_long']) & (self.data['SMA_short'].shift(1) >= self.data['SMA_long'].shift(1))

        signals['signal'] = np.where(signals['buy'], 1, np.where(signals['sell'], -1, 0))
        
        return signals['signal'].astype(int)

# -----------------------------------------------------------------------------
# Strategy 2: RSI Overbought/Oversold Strategy
# -----------------------------------------------------------------------------
class Strategy2_RSIOverboughtOversoldStrategy(StrategyBase):
    """
    A mean-reversion strategy using the Relative Strength Index (RSI).
    - Buy signal when RSI crosses above the oversold threshold.
    - Sell signal when RSI crosses below the overbought threshold.
    """
    def __init__(self, data, rsi_period=14, overbought_threshold=70, oversold_threshold=30):
        super().__init__(data)
        self.rsi_period = rsi_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

    def generate_signals(self):
        # Calculate RSI
        self.data['RSI'] = ta.rsi(self.data['Close'], length=self.rsi_period)
        
        # Conditions for buy and sell signals
        buy_condition = (self.data['RSI'].shift(1) < self.oversold_threshold) & (self.data['RSI'] >= self.oversold_threshold)
        sell_condition = (self.data['RSI'].shift(1) > self.overbought_threshold) & (self.data['RSI'] <= self.overbought_threshold)

        # Generate signals: 1 for buy, -1 for sell, 0 for hold
        signals = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))
        
        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 3: Bollinger Band Breakout Strategy
# -----------------------------------------------------------------------------
class Strategy3_BollingerBandBreakoutStrategy(StrategyBase):
    """
    A volatility breakout strategy using Bollinger Bands.
    - Buy signal when the closing price breaks above the upper Bollinger Band.
    - Sell signal when the closing price breaks below the lower Bollinger Band.
    """
    def __init__(self, data, length=20, std_dev=2.0):
        super().__init__(data)
        self.length = length
        self.std_dev = std_dev

    def generate_signals(self):
        # Calculate Bollinger Bands
        bbands = ta.bbands(self.data['Close'], length=self.length, std=self.std_dev)
        self.data['BBL'] = bbands[f'BBL_{self.length}_{self.std_dev}']
        self.data['BBU'] = bbands[f'BBU_{self.length}_{self.std_dev}']

        # Conditions for buy and sell signals
        buy_condition = (self.data['Close'].shift(1) <= self.data['BBU'].shift(1)) & (self.data['Close'] > self.data['BBU'])
        sell_condition = (self.data['Close'].shift(1) >= self.data['BBL'].shift(1)) & (self.data['Close'] < self.data['BBL'])

        # Generate signals
        signals = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))

        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 4: MACD Signal Cross Strategy
# -----------------------------------------------------------------------------
class Strategy4_MACDSignalCrossStrategy(StrategyBase):
    """
    Momentum strategy based on the MACD line crossing its signal line.
    - Buy signal when the MACD line crosses above the signal line.
    - Sell signal when the MACD line crosses below the signal line.
    """
    def __init__(self, data, fast=12, slow=26, signal=9):
        super().__init__(data)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self):
        # Calculate MACD
        macd = ta.macd(self.data['Close'], fast=self.fast, slow=self.slow, signal=self.signal)
        self.data['MACD'] = macd[f'MACD_{self.fast}_{self.slow}_{self.signal}']
        self.data['MACD_signal'] = macd[f'MACDs_{self.fast}_{self.slow}_{self.signal}']
        
        # Conditions for buy and sell signals
        buy_condition = (self.data['MACD'].shift(1) < self.data['MACD_signal'].shift(1)) & (self.data['MACD'] >= self.data['MACD_signal'])
        sell_condition = (self.data['MACD'].shift(1) > self.data['MACD_signal'].shift(1)) & (self.data['MACD'] <= self.data['MACD_signal'])
        
        # Generate signals
        signals = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))

        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 5: ADX Trend Strategy
# -----------------------------------------------------------------------------
class Strategy5_ADXTrendStrategy(StrategyBase):
    """
    Trend-following strategy using the Average Directional Index (ADX) to filter for trend strength.
    - A buy signal is generated if ADX is above a threshold (e.g., 25) and the +DI crosses above the -DI.
    - A sell signal is generated if ADX is above the threshold and the -DI crosses above the +DI.
    """
    def __init__(self, data, adx_period=14, adx_threshold=25):
        super().__init__(data)
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def generate_signals(self):
        # Calculate ADX
        adx = ta.adx(self.data['High'], self.data['Low'], self.data['Close'], length=self.adx_period)
        self.data['ADX'] = adx[f'ADX_{self.adx_period}']
        self.data['DMP'] = adx[f'DMP_{self.adx_period}'] # +DI
        self.data['DMN'] = adx[f'DMN_{self.adx_period}'] # -DI
        
        # Check if trend is strong enough
        strong_trend = self.data['ADX'] > self.adx_threshold
        
        # Conditions for buy and sell signals
        buy_crossover = (self.data['DMP'].shift(1) < self.data['DMN'].shift(1)) & (self.data['DMP'] >= self.data['DMN'])
        sell_crossover = (self.data['DMN'].shift(1) < self.data['DMP'].shift(1)) & (self.data['DMN'] >= self.data['DMP'])
        
        buy_condition = strong_trend & buy_crossover
        sell_condition = strong_trend & sell_crossover

        # Generate signals
        signals = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))

        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 6: Stochastic Oscillator Strategy
# -----------------------------------------------------------------------------
class Strategy6_StochasticOscillatorStrategy(StrategyBase):
    """
    Momentum strategy using the Stochastic Oscillator.
    - Buy when oscillator is oversold and %K crosses above %D.
    - Sell when oscillator is overbought and %K crosses below %D.
    """
    def __init__(self, data, k=14, d=3, smooth_k=3, oversold=20, overbought=80):
        super().__init__(data)
        self.k = k
        self.d = d
        self.smooth_k = smooth_k
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self):
        # Calculate Stochastic Oscillator
        stoch = ta.stoch(self.data['High'], self.data['Low'], self.data['Close'], k=self.k, d=self.d, smooth_k=self.smooth_k)
        self.data[f'STOCHk_{self.k}_{self.d}_{self.smooth_k}'] = stoch[f'STOCHk_{self.k}_{self.d}_{self.smooth_k}']
        self.data[f'STOCHd_{self.k}_{self.d}_{self.smooth_k}'] = stoch[f'STOCHd_{self.k}_{self.d}_{self.smooth_k}']

        # Conditions for buy and sell signals
        buy_condition = ((self.data[f'STOCHk_{self.k}_{self.d}_{self.smooth_k}'].shift(1) < self.oversold) & 
                         (self.data[f'STOCHk_{self.k}_{self.d}_{self.smooth_k}'] >= self.oversold))
        
        sell_condition = ((self.data[f'STOCHk_{self.k}_{self.d}_{self.smooth_k}'].shift(1) > self.overbought) & 
                          (self.data[f'STOCHk_{self.k}_{self.d}_{self.smooth_k}'] <= self.overbought))

        # Generate signals
        signals = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))

        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 7: Ichimoku Cloud Strategy
# -----------------------------------------------------------------------------

class Strategy7_IchimokuCloudStrategy(StrategyBase):
    """
    A trend-following system using the Ichimoku Cloud.
    - Buy when the price is above the Kumo Cloud and the Tenkan-sen crosses above the Kijun-sen.
    - Sell when the price is below the Kumo Cloud and the Tenkan-sen crosses below the Kijun-sen.
    """
    def __init__(self, data, tenkan=9, kijun=26, senkou=52):
        super().__init__(data)
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou = senkou

    def generate_signals(self):
        # Calculate Ichimoku Cloud components
        ichimoku_df, _ = ta.ichimoku(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            tenkan=self.tenkan,
            kijun=self.kijun,
            senkou=self.senkou
        )

        self.data['TENKAN'] = ichimoku_df[f'ITS_{self.tenkan}']
        self.data['KIJUN'] = ichimoku_df[f'IKS_{self.kijun}']
        self.data['SPAN_A'] = ichimoku_df[f'ISA_{self.tenkan}']
        self.data['SPAN_B'] = ichimoku_df[f'ISB_{self.kijun}']

        # Conditions for buy signal
        above_cloud = (self.data['Close'] > self.data['SPAN_A']) & (self.data['Close'] > self.data['SPAN_B'])
        tk_cross_up = (self.data['TENKAN'] > self.data['KIJUN']) & (self.data['TENKAN'].shift(1) <= self.data['KIJUN'].shift(1))
        buy_condition = above_cloud & tk_cross_up

        # Conditions for sell signal
        below_cloud = (self.data['Close'] < self.data['SPAN_A']) & (self.data['Close'] < self.data['SPAN_B'])
        tk_cross_down = (self.data['TENKAN'] < self.data['KIJUN']) & (self.data['TENKAN'].shift(1) >= self.data['KIJUN'].shift(1))
        sell_condition = below_cloud & tk_cross_down

        # Generate signals
        signals = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))
        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 8: Volume Spike MA Strategy
# -----------------------------------------------------------------------------
class Strategy8_VolumeSpikeMAStrategy(StrategyBase):
    """
    Combines a long-term moving average for trend direction with volume spikes
    to signal conviction in a move.
    - Buy: Price is above the 200-day SMA and Volume is > 2x its 50-day average.
    - Sell: Price is below the 200-day SMA and Volume is > 2x its 50-day average.
    """
    def __init__(self, data, ma_period=200, vol_ma_period=50, vol_spike_factor=2.0):
        super().__init__(data)
        self.ma_period = ma_period
        self.vol_ma_period = vol_ma_period
        self.vol_spike_factor = vol_spike_factor

    def generate_signals(self):
        # Calculate long-term SMA for trend and moving average for volume
        self.data['SMA'] = ta.sma(self.data['Close'], length=self.ma_period)
        self.data['Volume_SMA'] = ta.sma(self.data['Volume'], length=self.vol_ma_period)
        
        # Condition for a volume spike
        volume_spike = self.data['Volume'] > (self.data['Volume_SMA'] * self.vol_spike_factor)
        
        # Conditions for buy and sell signals
        buy_condition = (self.data['Close'] > self.data['SMA']) & volume_spike
        sell_condition = (self.data['Close'] < self.data['SMA']) & volume_spike

        # Generate signals
        signals = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))

        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 9: PSAR and EMA Trend Strategy
# -----------------------------------------------------------------------------
class Strategy9_PSAR_EMA_Strategy(StrategyBase):
    """
    Uses a Parabolic SAR for entry/exit signals, filtered by a long-term EMA
    to ensure trades are taken in the direction of the primary trend.
    - Buy: Price > 200-day EMA and PSAR flips below the price.
    - Sell: Price < 200-day EMA and PSAR flips above the price.
    """
    def __init__(self, data, ema_period=200, af=0.02, max_af=0.2):
        super().__init__(data)
        self.ema_period = ema_period
        self.af = af
        self.max_af = max_af

    def generate_signals(self):
        # Calculate long-term EMA and Parabolic SAR
        self.data['EMA'] = ta.ema(self.data['Close'], length=self.ema_period)
        psar = ta.psar(self.data['High'], self.data['Low'], af=self.af, max_af=self.max_af)
        self.data['PSAR'] = psar['PSARl_0.02_0.2'] # Use long PSAR column

        # Determine PSAR trend direction
        psar_uptrend = self.data['PSAR'] < self.data['Close']
        psar_downtrend = self.data['PSAR'] > self.data['Close']

        # Signal is generated on the first bar of a new PSAR trend
        buy_flip = psar_uptrend & ~psar_uptrend.shift(1).fillna(False)
        sell_flip = psar_downtrend & ~psar_downtrend.shift(1).fillna(False)

        # Filter signals with the long-term EMA trend
        buy_condition = (self.data['Close'] > self.data['EMA']) & buy_flip
        sell_condition = (self.data['Close'] < self.data['EMA']) & sell_flip

        # Generate signals
        signals = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))

        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 10: Awesome Oscillator Zero-Cross Strategy
# -----------------------------------------------------------------------------
class Strategy10_AwesomeOscillatorStrategy(StrategyBase):
    """
    A momentum strategy based on the Awesome Oscillator crossing the zero line.
    - Buy signal when the AO crosses from negative to positive.
    - Sell signal when the AO crosses from positive to negative.
    """
    def __init__(self, data, fast=5, slow=34):
        super().__init__(data)
        self.fast = fast
        self.slow = slow

    def generate_signals(self):
        # Calculate Awesome Oscillator
        self.data['AO'] = ta.ao(self.data['High'], self.data['Low'], fast=self.fast, slow=self.slow)

        # Conditions for buy and sell signals
        buy_condition = (self.data['AO'] > 0) & (self.data['AO'].shift(1) <= 0)
        sell_condition = (self.data['AO'] < 0) & (self.data['AO'].shift(1) >= 0)

        # Generate signals
        signals = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))
        
        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 11: Triple SuperTrend Strategy
# -----------------------------------------------------------------------------
class Strategy11_TripleSuperTrendStrategy(StrategyBase):
    """
    This strategy uses three SuperTrend indicators with different parameters.
    A buy signal is generated only when all three SuperTrend indicators are bullish (green).
    A sell signal is generated only when all three SuperTrend indicators are bearish (red).
    Reference: Page 8 of the provided document. 
    """
    def __init__(self, data, st_periods=[(12, 3.0), (10, 1.0), (11, 2.0)]):
        super().__init__(data)
        self.st_periods = st_periods

    def generate_signals(self):
        # Calculate the three SuperTrend indicators
        st1 = ta.supertrend(self.data['High'], self.data['Low'], self.data['Close'], length=self.st_periods[0][0], multiplier=self.st_periods[0][1])
        st2 = ta.supertrend(self.data['High'], self.data['Low'], self.data['Close'], length=self.st_periods[1][0], multiplier=self.st_periods[1][1])
        st3 = ta.supertrend(self.data['High'], self.data['Low'], self.data['Close'], length=self.st_periods[2][0], multiplier=self.st_periods[2][1])
        
        self.data['ST1_Trend'] = st1[f'SUPERTd_{self.st_periods[0][0]}_{self.st_periods[0][1]}']
        self.data['ST2_Trend'] = st2[f'SUPERTd_{self.st_periods[1][0]}_{self.st_periods[1][1]}']
        self.data['ST3_Trend'] = st3[f'SUPERTd_{self.st_periods[2][0]}_{self.st_periods[2][1]}']

        # Determine buy and sell conditions
        buy_condition = (self.data['ST1_Trend'] == 1) & (self.data['ST2_Trend'] == 1) & (self.data['ST3_Trend'] == 1)
        sell_condition = (self.data['ST1_Trend'] == -1) & (self.data['ST2_Trend'] == -1) & (self.data['ST3_Trend'] == -1)
        
        # Generate signals based on the first day all conditions are met
        signals = np.where(buy_condition & ~buy_condition.shift(1).fillna(False), 1, 0)
        signals = np.where(sell_condition & ~sell_condition.shift(1).fillna(False), -1, signals)

        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 12: Ichimoku SuperTrend Strategy
# -----------------------------------------------------------------------------
class Strategy12_IchimokuSuperTrendStrategy(StrategyBase):
    """
    Combines the SuperTrend indicator with the Ichimoku Cloud (Kumo).
    - Buy Signal: SuperTrend is bullish (green) AND the price is above the Kumo Cloud.
    - Sell Signal: SuperTrend is bearish (red) AND the price is below the Kumo Cloud.
    Reference: Page 9 of the provided document.
    """
    def __init__(self, data, st_length=10, st_multiplier=3.0, tenkan=9, kijun=26, senkou=52):
        super().__init__(data)
        self.st_length = st_length
        self.st_multiplier = st_multiplier
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou = senkou

    def generate_signals(self):
        # Calculate SuperTrend
        st = ta.supertrend(
            self.data['High'],
            self.data['Low'],
            self.data['Close'],
            length=self.st_length,
            multiplier=self.st_multiplier
        )
        self.data['ST_Trend'] = st[f'SUPERTd_{self.st_length}_{self.st_multiplier}']

        # Calculate Ichimoku Cloud
        ichimoku_df, _ = ta.ichimoku(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            tenkan=self.tenkan,
            kijun=self.kijun,
            senkou=self.senkou
        )

        self.data['SPAN_A'] = ichimoku_df[f'ISA_{self.tenkan}']
        self.data['SPAN_B'] = ichimoku_df[f'ISB_{self.kijun}']

        # Conditions for buy and sell signals
        buy_condition = (
            (self.data['ST_Trend'] == 1) &
            (self.data['Close'] > self.data['SPAN_A']) &
            (self.data['Close'] > self.data['SPAN_B'])
        )
        sell_condition = (
            (self.data['ST_Trend'] == -1) &
            (self.data['Close'] < self.data['SPAN_A']) &
            (self.data['Close'] < self.data['SPAN_B'])
        )

        # Generate signals on the first day the condition is met
        signals = np.where(buy_condition & ~buy_condition.shift(1).fillna(False), 1, 0)
        signals = np.where(sell_condition & ~sell_condition.shift(1).fillna(False), -1, signals)

        return pd.Series(signals, index=self.data.index, dtype=int)


# -----------------------------------------------------------------------------
# Strategy 13: Heikin Ashi Stochastic Strategy
# -----------------------------------------------------------------------------
class Strategy13_HeikinAshiStochasticStrategy(StrategyBase):
    """
    This strategy uses Heikin Ashi candles to identify a weak downtrend (pullback in an uptrend)
    and the Stochastic oscillator to time the entry.
    - Buy Signal: Stochastic crosses below oversold level during a weak downtrend identified by Heikin Ashi patterns.
    Reference: Page 6 of the provided document.
    """
    def __init__(self, data, k=14, d=3, smooth_k=3, oversold=20):
        super().__init__(data)
        self.k = k
        self.d = d
        self.smooth_k = smooth_k
        self.oversold = oversold

    def generate_signals(self):
        # Get Heikin Ashi self.data
        ha_data = ta.ha(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        
        # Calculate Stochastic Oscillator on original self.data
        stoch = ta.stoch(self.data['High'], self.data['Low'], self.data['Close'], k=self.k, d=self.d, smooth_k=self.smooth_k)
        stoch_k = stoch[f'STOCHk_{self.k}_{self.d}_{self.smooth_k}']

        # Identify weak downtrend with Heikin Ashi (e.g., small red bodies, upper wicks)
        # This is simplified to finding a doji candle after a downtrend
        is_doji = abs(ha_data['HA_close'] - ha_data['HA_open']) < (ha_data['HA_high'] - ha_data['HA_low']) * 0.1
        was_downtrend = ha_data['HA_close'].shift(1) < ha_data['HA_open'].shift(1)
        
        # Buy condition: stochastic crosses below oversold and a doji appears after a downtrend
        stoch_oversold_cross = (stoch_k.shift(1) > self.oversold) & (stoch_k <= self.oversold)
        buy_condition = stoch_oversold_cross & is_doji & was_downtrend

        signals = np.where(buy_condition, 1, 0)
        # Sell signal is not defined in the reference, so we will not implement it.
        
        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 14: Engulfing Candle EMA Strategy
# -----------------------------------------------------------------------------
class Strategy14_EngulfingCandleEMAStrategy(StrategyBase):
    """
    A trend-following strategy that uses a long-term EMA to determine the trend
    and a bullish engulfing candle for a buy entry signal.
    - Buy Signal: Price is above the 200-day EMA and a bullish engulfing candle appears.
    Reference: Page 12 of the provided document.
    """
    def __init__(self, data, ema_period=200):
        super().__init__(data)
        self.ema_period = ema_period

    def generate_signals(self):
        # Calculate the trend-filtering EMA
        self.data['EMA_trend'] = ta.ema(self.data['Close'], length=self.ema_period)
        
        # Identify bullish engulfing candles
        engulfing = ta.cdl_pattern(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'], name="engulfing")
        is_bullish_engulfing = engulfing == 100

        # Condition for buy signal
        in_uptrend = self.data['Close'] > self.data['EMA_trend']
        buy_condition = in_uptrend & is_bullish_engulfing

        signals = np.where(buy_condition, 1, 0)
        # Sell signal is not explicitly defined in the reference in the same way.
        
        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 15: DEMA SuperTrend Strategy
# -----------------------------------------------------------------------------
class Strategy15_DEMA_SuperTrendStrategy(StrategyBase):
    """
    This strategy combines a 200-period Double Exponential Moving Average (DEMA)
    with the SuperTrend indicator.
    - Buy Signal: Price is above the DEMA and the SuperTrend indicator gives a buy signal.
    - Sell Signal: Price is below the DEMA and the SuperTrend indicator gives a sell signal.
    Reference: Pages 13-14 of the provided document. 
    """
    def __init__(self, data, dema_period=200, st_length=12, st_multiplier=3.0):
        super().__init__(data)
        self.dema_period = dema_period
        self.st_length = st_length
        self.st_multiplier = st_multiplier

    def generate_signals(self):
        # Calculate DEMA
        self.data['DEMA'] = ta.dema(self.data['Close'], length=self.dema_period)
        
        # Calculate SuperTrend
        st = ta.supertrend(self.data['High'], self.data['Low'], self.data['Close'], length=self.st_length, multiplier=self.st_multiplier)
        self.data['ST_Buy_Signal'] = st[f'SUPERTl_{self.st_length}_{self.st_multiplier}']
        self.data['ST_Sell_Signal'] = st[f'SUPERTs_{self.st_length}_{self.st_multiplier}']

        # Conditions
        buy_entry = (self.data['Close'] > self.data['DEMA']) & (self.data['ST_Buy_Signal'].notna())
        sell_entry = (self.data['Close'] < self.data['DEMA']) & (self.data['ST_Sell_Signal'].notna())

        # Generate signals on the first day the condition is met
        signals = np.where(buy_entry & ~buy_entry.shift(1).fillna(False), 1, 0)
        signals = np.where(sell_entry & ~sell_entry.shift(1).fillna(False), -1, signals)
        
        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 16: Multi EMA Cross Strategy
# -----------------------------------------------------------------------------
class Strategy16_MultiEMACrossStrategy(StrategyBase):
    """
    Uses a ribbon of four EMAs (8, 13, 21, 55).
    - Buy Signal: The 55 EMA is at the bottom of the ribbon (all faster EMAs are above it).
    - Sell Signal: The 55 EMA is at the top of the ribbon (all faster EMAs are below it).
    Reference: Page 18 of the provided document. 
    """
    def __init__(self, data, lengths=[8, 13, 21, 55]):
        super().__init__(data)
        self.lengths = sorted(lengths)

    def generate_signals(self):
        # Calculate all EMAs
        for length in self.lengths:
            self.data[f'EMA_{length}'] = ta.ema(self.data['Close'], length=length)
        
        # Buy condition: longest EMA is the lowest value
        buy_condition = self.data[f'EMA_{self.lengths[0]}'] > self.data[f'EMA_{self.lengths[1]}']
        for i in range(1, len(self.lengths) - 1):
             buy_condition &= self.data[f'EMA_{self.lengths[i]}'] > self.data[f'EMA_{self.lengths[i+1]}']
        
        # Sell condition: longest EMA is the highest value
        sell_condition = self.data[f'EMA_{self.lengths[0]}'] < self.data[f'EMA_{self.lengths[1]}']
        for i in range(1, len(self.lengths) - 1):
             sell_condition &= self.data[f'EMA_{self.lengths[i]}'] < self.data[f'EMA_{self.lengths[i+1]}']

        # Generate signals on first day of condition
        signals = np.where(buy_condition & ~buy_condition.shift(1).fillna(False), 1, 0)
        signals = np.where(sell_condition & ~sell_condition.shift(1).fillna(False), -1, signals)

        return pd.Series(signals, index=self.data.index, dtype=int)
        
# -----------------------------------------------------------------------------
# Strategy 17: SSL CMF Strategy
# -----------------------------------------------------------------------------
class Strategy17_SSL_CMF_Strategy(StrategyBase):
    """
    Combines the SSL Channel with the Chaikin Money Flow (CMF).
    - Buy Signal: Bullish cross on the SSL Channel and CMF is positive.
    - Sell Signal: Bearish cross on the SSL Channel and CMF is negative.
    Reference: Pages 27-28 of the provided document.
    """
    def __init__(self, data, ssl_period=20, cmf_period=20):
        super().__init__(data)
        self.ssl_period = ssl_period
        self.cmf_period = cmf_period

    def generate_signals(self):
        # Calculate SSL Channel
        high_ma = ta.sma(self.data['High'], length=self.ssl_period)
        low_ma = ta.sma(self.data['Low'], length=self.ssl_period)
        self.data['SSL_Up'] = np.where(self.data['Close'] > high_ma, high_ma, low_ma)
        self.data['SSL_Down'] = np.where(self.data['Close'] < low_ma, low_ma, high_ma)
        
        # Calculate CMF
        self.data['CMF'] = ta.cmf(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'], length=self.cmf_period)
        
        # SSL Cross conditions
        buy_cross = (self.data['Close'] > self.data['SSL_Down']) & (self.data['Close'].shift(1) <= self.data['SSL_Down'].shift(1))
        sell_cross = (self.data['Close'] < self.data['SSL_Up']) & (self.data['Close'].shift(1) >= self.data['SSL_Up'].shift(1))
        
        # Final conditions with CMF filter
        buy_condition = buy_cross & (self.data['CMF'] > 0)
        sell_condition = sell_cross & (self.data['CMF'] < 0)

        signals = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))

        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 18: QQE HullSuite Strategy
# -----------------------------------------------------------------------------
class Strategy18_QQE_HullSuiteStrategy(StrategyBase):
    """
    Uses the QQE Mod indicator and the Hull Suite moving average.
    - Buy Signal: QQE histogram is blue and price is above the Hull MA.
    - Sell Signal: QQE histogram is red and price is below the Hull MA.
    Reference: Page 28 of the provided document.
    """
    def __init__(self, data, hull_length=60, qqe_length=14):
        super().__init__(data)
        self.hull_length = hull_length
        self.qqe_length = qqe_length

    def generate_signals(self):
        # Calculate Hull Moving Average
        self.data['HMA'] = ta.hma(self.data['Close'], length=self.hull_length)
        
        # Calculate QQE Mod (using a simplified proxy with RSI and its smoothed version)
        rsi = ta.rsi(self.data['Close'], length=self.qqe_length)
        smoothed_rsi = ta.ema(rsi, length=self.qqe_length)
        
        # Conditions
        qqe_bullish = rsi > smoothed_rsi
        qqe_bearish = rsi < smoothed_rsi
        
        price_above_hma = self.data['Close'] > self.data['HMA']
        price_below_hma = self.data['Close'] < self.data['HMA']
        
        buy_condition = qqe_bullish & price_above_hma
        sell_condition = qqe_bearish & price_below_hma

        # Generate signals on the first day
        signals = np.where(buy_condition & ~buy_condition.shift(1).fillna(False), 1, 0)
        signals = np.where(sell_condition & ~sell_condition.shift(1).fillna(False), -1, signals)

        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 19: Fair Value Gap Entry Strategy
# -----------------------------------------------------------------------------
class Strategy19_FairValueGapEntryStrategy(StrategyBase):
    """
    Identifies Fair Value Gaps (FVG) and enters on a retracement.
    - Bearish FVG: Low of candle[n] is below High of candle[n-2]. Gap is between Low[n] and High[n-2].
    - Bullish FVG: High of candle[n] is above Low of candle[n-2]. Gap is between High[n] and Low[n-2].
    Signal is generated when price enters this gap.
    Reference: Page 35 of the provided document. 
    """
    def generate_signals(self):
        # Identify Bearish Fair Value Gap
        bearish_fvg_high = self.data['High'].shift(2)
        bearish_fvg_low = self.data['Low']
        is_bearish_fvg = bearish_fvg_low > bearish_fvg_high
        
        # Identify Bullish Fair Value Gap
        bullish_fvg_low = self.data['Low'].shift(2)
        bullish_fvg_high = self.data['High']
        is_bullish_fvg = bullish_fvg_high < bullish_fvg_low

        # Entry Conditions (price retraces into the gap)
        enter_bearish_fvg = is_bearish_fvg & (self.data['High'] >= bearish_fvg_high)
        enter_bullish_fvg = is_bullish_fvg & (self.data['Low'] <= bullish_fvg_low)

        signals = np.where(enter_bullish_fvg, 1, np.where(enter_bearish_fvg, -1, 0))

        return pd.Series(signals, index=self.data.index, dtype=int)

# -----------------------------------------------------------------------------
# Strategy 20: Liquidity Grab Reversal Strategy
# -----------------------------------------------------------------------------
class Strategy20_LiquidityGrabReversalStrategy(StrategyBase):
    """
    Identifies a "liquidity grab" or "stop hunt" and plays the reversal.
    - Buy Signal: Price makes a new N-day low but then closes above that low and the previous day's low.
    - Sell Signal: Price makes a new N-day high but then closes below that high and the previous day's high.
    Reference: Page 43 of the provided document.
    """
    def __init__(self, data, period=20):
        super().__init__(data)
        self.period = period

    def generate_signals(self):
        self.data['N_day_low'] = self.data['Low'].shift(1).rolling(window=self.period).min()
        self.data['N_day_high'] = self.data['High'].shift(1).rolling(window=self.period).max()
        
        # Bullish Liquidity Grab (Stop Hunt Low)
        grab_low = self.data['Low'] < self.data['N_day_low']
        reversal_close_up = self.data['Close'] > self.data['Low'].shift(1)
        buy_condition = grab_low & reversal_close_up
        
        # Bearish Liquidity Grab (Stop Hunt High)
        grab_high = self.data['High'] > self.data['N_day_high']
        reversal_close_down = self.data['Close'] < self.data['High'].shift(1)
        sell_condition = grab_high & reversal_close_down

        signals = np.where(buy_condition, 1, np.where(sell_condition, -1, 0))
        
        return pd.Series(signals, index=self.data.index, dtype=int)

# --------------------------------------------------------------------------------
# Strategy Implementations (21 to 40)
# --------------------------------------------------------------------------------

class Strategy21_SMACrossover(StrategyBase):
    """
    Strategy 21: Simple Moving Average (SMA) Crossover
    - Buy Signal: When the short-term SMA (e.g., 50-day) crosses above the long-term SMA (e.g., 200-day).
    - Sell Signal: When the short-term SMA crosses below the long-term SMA.
    """
    def generate_signals(self):
        # Calculate SMAs
        self.data['SMA_50'] = ta.sma(self.data['Close'], length=50)
        self.data['SMA_200'] = ta.sma(self.data['Close'], length=200)

        # Initialize signals column
        signals = pd.Series(0, index=self.data.index, dtype=int)

        # Generate Buy signal (Golden Cross)
        buy_condition = (self.data['SMA_50'] > self.data['SMA_200']) & (self.data['SMA_50'].shift(1) <= self.data['SMA_200'].shift(1))
        signals[buy_condition] = 1

        # Generate Sell signal (Death Cross)
        sell_condition = (self.data['SMA_50'] < self.data['SMA_200']) & (self.data['SMA_50'].shift(1) >= self.data['SMA_200'].shift(1))
        signals[sell_condition] = -1

        return signals

class Strategy22_EMACrossover(StrategyBase):
    """
    Strategy 22: Exponential Moving Average (EMA) Crossover
    - Buy Signal: When the short-term EMA (e.g., 12-day) crosses above the long-term EMA (e.g., 26-day).
    - Sell Signal: When the short-term EMA crosses below the long-term EMA.
    """
    def generate_signals(self):
        self.data['EMA_12'] = ta.ema(self.data['Close'], length=12)
        self.data['EMA_26'] = ta.ema(self.data['Close'], length=26)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['EMA_12'] > self.data['EMA_26']) & (self.data['EMA_12'].shift(1) <= self.data['EMA_26'].shift(1))
        sell_condition = (self.data['EMA_12'] < self.data['EMA_26']) & (self.data['EMA_12'].shift(1) >= self.data['EMA_26'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy23_MACDStrategy(StrategyBase):
    """
    Strategy 23: MACD Crossover
    - Buy Signal: When the MACD line crosses above the MACD signal line.
    - Sell Signal: When the MACD line crosses below the MACD signal line.
    """
    def generate_signals(self):
        macd = ta.macd(self.data['Close'], fast=12, slow=26, signal=9)
        self.data['MACD'] = macd['MACD_12_26_9']
        self.data['MACD_signal'] = macd['MACDs_12_26_9']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['MACD'] > self.data['MACD_signal']) & (self.data['MACD'].shift(1) <= self.data['MACD_signal'].shift(1))
        sell_condition = (self.data['MACD'] < self.data['MACD_signal']) & (self.data['MACD'].shift(1) >= self.data['MACD_signal'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy24_RSIReversal(StrategyBase):
    """
    Strategy 24: RSI Mean Reversion
    - Buy Signal: When RSI crosses below an oversold threshold (e.g., 30).
    - Sell Signal: When RSI crosses above an overbought threshold (e.g., 70).
    """
    def generate_signals(self):
        self.data['RSI'] = ta.rsi(self.data['Close'], length=14)
        oversold_threshold = 30
        overbought_threshold = 70

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['RSI'] < oversold_threshold) & (self.data['RSI'].shift(1) >= oversold_threshold)
        sell_condition = (self.data['RSI'] > overbought_threshold) & (self.data['RSI'].shift(1) <= overbought_threshold)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy25_BollingerBreakout(StrategyBase):
    """
    Strategy 25: Bollinger Bands Breakout (Trend Following)
    - Buy Signal: When the closing price crosses above the upper Bollinger Band.
    - Sell Signal: When the closing price crosses below the lower Bollinger Band.
    """
    def generate_signals(self):
        bbands = ta.bbands(self.data['Close'], length=20, std=2)
        self.data['BB_upper'] = bbands['BBU_20_2.0']
        self.data['BB_lower'] = bbands['BBL_20_2.0']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] > self.data['BB_upper']) & (self.data['Close'].shift(1) <= self.data['BB_upper'].shift(1))
        sell_condition = (self.data['Close'] < self.data['BB_lower']) & (self.data['Close'].shift(1) >= self.data['BB_lower'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy26_BollingerReversion(StrategyBase):
    """
    Strategy 26: Bollinger Bands Mean Reversion
    - Buy Signal: When the closing price crosses below the lower Bollinger Band.
    - Sell Signal: When the closing price crosses above the upper Bollinger Band.
    """
    def generate_signals(self):
        bbands = ta.bbands(self.data['Close'], length=20, std=2)
        self.data['BB_upper'] = bbands['BBU_20_2.0']
        self.data['BB_lower'] = bbands['BBL_20_2.0']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        # Note: The signals are opposite of the breakout strategy.
        buy_condition = (self.data['Close'] < self.data['BB_lower'])
        sell_condition = (self.data['Close'] > self.data['BB_upper'])

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy27_StochasticCrossover(StrategyBase):
    """
    Strategy 27: Stochastic Oscillator Crossover
    - Buy Signal: When %K line crosses above the %D line in the oversold region (e.g., below 20).
    - Sell Signal: When %K line crosses below the %D line in the overbought region (e.g., above 80).
    """
    def generate_signals(self):
        stoch = ta.stoch(self.data['High'], self.data['Low'], self.data['Close'], k=14, d=3, smooth_k=3)
        self.data['STOCH_K'] = stoch['STOCHk_14_3_3']
        self.data['STOCH_D'] = stoch['STOCHd_14_3_3']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        oversold_level = 20
        overbought_level = 80

        buy_condition = (self.data['STOCH_K'] > self.data['STOCH_D']) & (self.data['STOCH_K'].shift(1) <= self.data['STOCH_D'].shift(1)) & (self.data['STOCH_D'] < oversold_level)
        sell_condition = (self.data['STOCH_K'] < self.data['STOCH_D']) & (self.data['STOCH_K'].shift(1) >= self.data['STOCH_D'].shift(1)) & (self.data['STOCH_D'] > overbought_level)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy28_ADXTrend(StrategyBase):
    """
    Strategy 28: ADX Trend Filter with EMA Crossover
    - Buy Signal: Short EMA crosses above Long EMA AND ADX is above a threshold (e.g., 25).
    - Sell Signal: Short EMA crosses below Long EMA AND ADX is above the threshold.
    """
    def generate_signals(self):
        adx = ta.adx(self.data['High'], self.data['Low'], self.data['Close'], length=14)
        self.data['ADX'] = adx['ADX_14']
        self.data['EMA_20'] = ta.ema(self.data['Close'], length=20)
        self.data['EMA_50'] = ta.ema(self.data['Close'], length=50)
        adx_threshold = 25

        signals = pd.Series(0, index=self.data.index, dtype=int)
        is_trending = self.data['ADX'] > adx_threshold

        buy_condition = (self.data['EMA_20'] > self.data['EMA_50']) & (self.data['EMA_20'].shift(1) <= self.data['EMA_50'].shift(1)) & is_trending
        sell_condition = (self.data['EMA_20'] < self.data['EMA_50']) & (self.data['EMA_20'].shift(1) >= self.data['EMA_50'].shift(1)) & is_trending

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy29_HeikinAshiTrend(StrategyBase):
    """
    Strategy 29: Heikin-Ashi Trend Following
    Uses Heikin-Ashi candles to identify the trend.
    - Buy Signal: Switch from a bearish (red) Heikin-Ashi candle to a bullish (green) one.
    - Sell Signal: Switch from a bullish Heikin-Ashi candle to a bearish one.
    """
    def generate_signals(self):
        ha_data = ta.ha(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        ha_close = ha_data['HA_close']
        ha_open = ha_data['HA_open']

        # A green candle is when HA_Close > HA_Open. A red candle is the opposite.
        is_green = ha_close > ha_open
        is_red = ha_close < ha_open

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = is_green & is_red.shift(1)
        sell_condition = is_red & is_green.shift(1)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy30_IchimokuBreakout(StrategyBase):
    """
    Strategy 30: Ichimoku Cloud Breakout
    - Buy Signal: Price closes above the Ichimoku Cloud (Senkou Span A and B).
    - Sell Signal: Price closes below the Ichimoku Cloud.
    """
    def __init__(self, data, tenkan=9, kijun=26, senkou=52):
        super().__init__(data)
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou = senkou

    def generate_signals(self):
        # Calculate Ichimoku Cloud components
        ichimoku_df, _ = ta.ichimoku(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            tenkan=self.tenkan,
            kijun=self.kijun,
            senkou=self.senkou
        )

        self.data['Senkou_A'] = ichimoku_df[f'ISA_{self.tenkan}']
        self.data['Senkou_B'] = ichimoku_df[f'ISB_{self.kijun}']

        # Initialize signal series
        signals = pd.Series(0, index=self.data.index, dtype=int)

        # Define breakout conditions
        above_cloud = (self.data['Close'] > self.data['Senkou_A']) & (self.data['Close'] > self.data['Senkou_B'])
        below_cloud = (self.data['Close'] < self.data['Senkou_A']) & (self.data['Close'] < self.data['Senkou_B'])

        buy_condition = above_cloud & ~above_cloud.shift(1, fill_value=False)
        sell_condition = below_cloud & ~below_cloud.shift(1, fill_value=False)

        # Assign signals
        signals[buy_condition] = 1
        signals[sell_condition] = -1

        return signals


class Strategy31_VwapCrossover(StrategyBase):
    """
    Strategy 31: Daily VWAP Crossover
    Since intraday self.data is not available, this uses a rolling VWAP.
    - Buy Signal: Close crosses above the rolling VWAP (e.g., 10-day).
    - Sell Signal: Close crosses below the rolling VWAP.
    """
    def generate_signals(self):
        self.data['VWAP'] = ta.vwap(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'], length=10)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] > self.data['VWAP']) & (self.data['Close'].shift(1) <= self.data['VWAP'].shift(1))
        sell_condition = (self.data['Close'] < self.data['VWAP']) & (self.data['Close'].shift(1) >= self.data['VWAP'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy32_DonchianBreakout(StrategyBase):
    """
    Strategy 32: Donchian Channel Breakout
    - Buy Signal: Close breaks above the upper Donchian channel (e.g., 20-day high).
    - Sell Signal: Close breaks below the lower Donchian channel (e.g., 20-day low).
    """
    def generate_signals(self):
        donchian = ta.donchian(self.data['High'], self.data['Low'], lower_length=20, upper_length=20)
        self.data['DC_upper'] = donchian['DCU_20_20']
        self.data['DC_lower'] = donchian['DCL_20_20']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] > self.data['DC_upper'].shift(1))
        sell_condition = (self.data['Close'] < self.data['DC_lower'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy33_KeltnerChannelBreakout(StrategyBase):
    """
    Strategy 33: Keltner Channel Breakout
    - Buy Signal: Price closes above the upper Keltner Channel band.
    - Sell Signal: Price closes below the lower Keltner Channel band.
    """
    def generate_signals(self):
        kc = ta.kc(self.data['High'], self.data['Low'], self.data['Close'], length=20, scalar=2.0)
        self.data['KC_upper'] = kc['KCUe_20_2.0']
        self.data['KC_lower'] = kc['KCLe_20_2.0']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] > self.data['KC_upper']) & (self.data['Close'].shift(1) <= self.data['KC_upper'].shift(1))
        sell_condition = (self.data['Close'] < self.data['KC_lower']) & (self.data['Close'].shift(1) >= self.data['KC_lower'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy34_AwesomeOscillatorZeroCross(StrategyBase):
    """
    Strategy 34: Awesome Oscillator (AO) Zero Line Crossover
    - Buy Signal: AO crosses above the zero line.
    - Sell Signal: AO crosses below the zero line.
    """
    def generate_signals(self):
        self.data['AO'] = ta.ao(self.data['High'], self.data['Low'], fast=5, slow=34)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['AO'] > 0) & (self.data['AO'].shift(1) <= 0)
        sell_condition = (self.data['AO'] < 0) & (self.data['AO'].shift(1) >= 0)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy35_ROCMomentum(StrategyBase):
    """
    Strategy 35: Rate of Change (ROC) Momentum
    - Buy Signal: ROC crosses above a certain positive level (e.g., 2).
    - Sell Signal: ROC crosses below a certain negative level (e.g., -2).
    """
    def generate_signals(self):
        self.data['ROC'] = ta.roc(self.data['Close'], length=12)
        upper_threshold = 2
        lower_threshold = -2

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['ROC'] > upper_threshold) & (self.data['ROC'].shift(1) <= upper_threshold)
        sell_condition = (self.data['ROC'] < lower_threshold) & (self.data['ROC'].shift(1) >= lower_threshold)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy36_TrixCrossover(StrategyBase):
    """
    Strategy 36: TRIX Signal Crossover
    - Buy Signal: TRIX line crosses above its signal line.
    - Sell Signal: TRIX line crosses below its signal line.
    """
    def generate_signals(self):
        trix = ta.trix(self.data['Close'], length=15, signal=9)
        self.data['TRIX'] = trix['TRIX_15_9']
        self.data['TRIX_signal'] = trix['TRIXs_15_9']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['TRIX'] > self.data['TRIX_signal']) & (self.data['TRIX'].shift(1) <= self.data['TRIX_signal'].shift(1))
        sell_condition = (self.data['TRIX'] < self.data['TRIX_signal']) & (self.data['TRIX'].shift(1) >= self.data['TRIX_signal'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy37_ChaikinMoneyFlow(StrategyBase):
    """
    Strategy 37: Chaikin Money Flow (CMF)
    - Buy Signal: CMF crosses above zero (indicating buying pressure).
    - Sell Signal: CMF crosses below zero (indicating selling pressure).
    """
    def generate_signals(self):
        self.data['CMF'] = ta.cmf(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'], length=20)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['CMF'] > 0) & (self.data['CMF'].shift(1) <= 0)
        sell_condition = (self.data['CMF'] < 0) & (self.data['CMF'].shift(1) >= 0)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy38_OBVTrend(StrategyBase):
    """
    Strategy 38: On-Balance Volume (OBV) Trend Confirmation
    - Buy Signal: Price is in an uptrend (Close > SMA 50) and OBV is also in an uptrend (OBV > its SMA).
    - Sell Signal: Price is in a downtrend (Close < SMA 50) and OBV is also in a downtrend.
    """
    def generate_signals(self):
        self.data['OBV'] = ta.obv(self.data['Close'], self.data['Volume'])
        self.data['OBV_SMA'] = ta.sma(self.data['OBV'], length=20)
        self.data['Price_SMA'] = ta.sma(self.data['Close'], length=50)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        price_uptrend = self.data['Close'] > self.data['Price_SMA']
        obv_uptrend = self.data['OBV'] > self.data['OBV_SMA']
        price_downtrend = self.data['Close'] < self.data['Price_SMA']
        obv_downtrend = self.data['OBV'] < self.data['OBV_SMA']

        # Signal is generated on the confirmation of the trend change
        buy_condition = price_uptrend & obv_uptrend & ~(price_uptrend.shift(1) & obv_uptrend.shift(1))
        sell_condition = price_downtrend & obv_downtrend & ~(price_downtrend.shift(1) & obv_downtrend.shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy39_SuperTrend(StrategyBase):
    """
    Strategy 39: SuperTrend Indicator
    - Buy Signal: SuperTrend indicator flips from showing a downtrend to an uptrend.
    - Sell Signal: SuperTrend indicator flips from showing an uptrend to a downtrend.
    """

    def generate_signals(self):
        supertrend = ta.supertrend(self.data['High'], self.data['Low'], self.data['Close'], length=10, multiplier=3)
        # SUPERTd_10_3.0 column is 1 for uptrend, -1 for downtrend
        self.data['ST_direction'] = supertrend['SUPERTd_10_3.0']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['ST_direction'] == 1) & (self.data['ST_direction'].shift(1) == -1)
        sell_condition = (self.data['ST_direction'] == -1) & (self.data['ST_direction'].shift(1) == 1)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy40_CoppockCurve(StrategyBase):
    """
    Strategy 40: Coppock Curve
    A long-term momentum indicator.
    - Buy Signal: Coppock Curve crosses above the zero line.
    - Sell Signal: Coppock Curve crosses below the zero line.
    """
    def generate_signals(self):
        self.data['Coppock'] = ta.coppock(self.data['Close'], length=10, fast=11, slow=14)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Coppock'] > 0) & (self.data['Coppock'].shift(1) <= 0)
        sell_condition = (self.data['Coppock'] < 0) & (self.data['Coppock'].shift(1) >= 0)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy41_RsiStochasticCombo(StrategyBase):
    """
    Strategy 41: RSI and Stochastic Combo
    - Buy Signal: RSI is oversold (< 30) AND Stochastic %K is also oversold (< 20).
    - Sell Signal: RSI is overbought (> 70) AND Stochastic %K is also overbought (> 80).
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['RSI'] = ta.rsi(self.data['Close'], length=14)
        stoch = ta.stoch(self.data['High'], self.data['Low'], self.data['Close'], k=14, d=3, smooth_k=3)
        self.data['STOCH_K'] = stoch['STOCHk_14_3_3']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        
        buy_condition = (self.data['RSI'] < 30) & (self.data['STOCH_K'] < 20)
        sell_condition = (self.data['RSI'] > 70) & (self.data['STOCH_K'] > 80)

        # Generate signal on the first day the condition is met
        signals[buy_condition & ~buy_condition.shift(1, fill_value=False)] = 1
        signals[sell_condition & ~sell_condition.shift(1, fill_value=False)] = -1
        
        return signals

class Strategy42_MacdRsiFilter(StrategyBase):
    """
    Strategy 42: MACD Crossover with RSI Momentum Filter
    - Buy Signal: MACD line crosses above its signal line, but only if RSI is above 50 (confirming bullish momentum).
    - Sell Signal: MACD line crosses below its signal line, but only if RSI is below 50 (confirming bearish momentum).
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        macd = ta.macd(self.data['Close'], fast=12, slow=26, signal=9)
        self.data['MACD'] = macd['MACD_12_26_9']
        self.data['MACD_signal'] = macd['MACDs_12_26_9']
        self.data['RSI'] = ta.rsi(self.data['Close'], length=14)

        signals = pd.Series(0, index=self.data.index, dtype=int)

        macd_cross_up = (self.data['MACD'] > self.data['MACD_signal']) & (self.data['MACD'].shift(1) <= self.data['MACD_signal'].shift(1))
        macd_cross_down = (self.data['MACD'] < self.data['MACD_signal']) & (self.data['MACD'].shift(1) >= self.data['MACD_signal'].shift(1))

        buy_condition = macd_cross_up & (self.data['RSI'] > 50)
        sell_condition = macd_cross_down & (self.data['RSI'] < 50)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy43_VwmacdCrossover(StrategyBase):
    """
    Strategy 43: Volume-Weighted MACD (VWmacd) Crossover
    - Buy Signal: VWmacd line crosses above its signal line.
    - Sell Signal: VWmacd line crosses below its signal line.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        vwmacd = ta.vwmacd(self.data['Close'], self.data['Volume'], fast=12, slow=26, signal=9)
        self.data['VWMACD'] = vwmacd['VWMACD_12_26_9']
        self.data['VWMACD_signal'] = vwmacd['VWMACDs_12_26_9']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['VWMACD'] > self.data['VWMACD_signal']) & (self.data['VWMACD'].shift(1) <= self.data['VWMACD_signal'].shift(1))
        sell_condition = (self.data['VWMACD'] < self.data['VWMACD_signal']) & (self.data['VWMACD'].shift(1) >= self.data['VWMACD_signal'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy44_ForceIndexZeroCross(StrategyBase):
    """
    Strategy 44: Elder's Force Index (EFI) Zero Crossover
    - Buy Signal: The smoothed Force Index (e.g., 13-period EMA) crosses above zero.
    - Sell Signal: The smoothed Force Index crosses below zero.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        # Calculate raw Force Index
        force_raw = (self.data['Close'] - self.data['Close'].shift(1)) * self.data['Volume']
        # Smooth with an EMA
        self.data['ForceIndex'] = ta.ema(force_raw, length=13)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['ForceIndex'] > 0) & (self.data['ForceIndex'].shift(1) <= 0)
        sell_condition = (self.data['ForceIndex'] < 0) & (self.data['ForceIndex'].shift(1) >= 0)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy45_WilliamsRReversal(StrategyBase):
    """
    Strategy 45: Williams %R (WillR) Reversal
    - Buy Signal: WillR crosses above the oversold level (e.g., -80).
    - Sell Signal: WillR crosses below the overbought level (e.g., -20).
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['WillR'] = ta.willr(self.data['High'], self.data['Low'], self.data['Close'], length=14)
        oversold_level = -80
        overbought_level = -20

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['WillR'] > oversold_level) & (self.data['WillR'].shift(1) <= oversold_level)
        sell_condition = (self.data['WillR'] < overbought_level) & (self.data['WillR'].shift(1) >= overbought_level)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy46_TemaPriceCross(StrategyBase):
    """
    Strategy 46: Triple Exponential Moving Average (TEMA) Price Crossover
    - Buy Signal: The closing price crosses above the TEMA.
    - Sell Signal: The closing price crosses below the TEMA.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['TEMA'] = ta.tema(self.data['Close'], length=20)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] > self.data['TEMA']) & (self.data['Close'].shift(1) <= self.data['TEMA'].shift(1))
        sell_condition = (self.data['Close'] < self.data['TEMA']) & (self.data['Close'].shift(1) >= self.data['TEMA'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy47_DemaCrossover(StrategyBase):
    """
    Strategy 47: Double EMA (DEMA) Crossover
    - Buy Signal: Fast DEMA (e.g., 10-period) crosses above Slow DEMA (e.g., 30-period).
    - Sell Signal: Fast DEMA crosses below Slow DEMA.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['DEMA_fast'] = ta.dema(self.data['Close'], length=10)
        self.data['DEMA_slow'] = ta.dema(self.data['Close'], length=30)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['DEMA_fast'] > self.data['DEMA_slow']) & (self.data['DEMA_fast'].shift(1) <= self.data['DEMA_slow'].shift(1))
        sell_condition = (self.data['DEMA_fast'] < self.data['DEMA_slow']) & (self.data['DEMA_fast'].shift(1) >= self.data['DEMA_slow'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy48_CmoReversal(StrategyBase):
    """
    Strategy 48: Chande Momentum Oscillator (CMO) Reversal
    - Buy Signal: CMO crosses above its oversold line (e.g., -50).
    - Sell Signal: CMO crosses below its overbought line (e.g., 50).
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['CMO'] = ta.cmo(self.data['Close'], length=14)
        oversold_level = -50
        overbought_level = 50

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['CMO'] > oversold_level) & (self.data['CMO'].shift(1) <= oversold_level)
        sell_condition = (self.data['CMO'] < overbought_level) & (self.data['CMO'].shift(1) >= overbought_level)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy49_AroonOscillatorCross(StrategyBase):
    """
    Strategy 49: Aroon Oscillator Zero Crossover
    - Buy Signal: The Aroon Oscillator crosses above zero.
    - Sell Signal: The Aroon Oscillator crosses below zero.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        aroon = ta.aroon(self.data['High'], self.data['Low'], length=25)
        self.data['Aroon_Osc'] = aroon['AROONOSC_25']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Aroon_Osc'] > 0) & (self.data['Aroon_Osc'].shift(1) <= 0)
        sell_condition = (self.data['Aroon_Osc'] < 0) & (self.data['Aroon_Osc'].shift(1) >= 0)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy50_ParabolicSarFlip(StrategyBase):
    """
    Strategy 50: Parabolic SAR (PSAR) Trend Flip
    - Buy Signal: PSAR flips from being above the price to below the price.
    - Sell Signal: PSAR flips from being below the price to above the price.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        psar = ta.psar(self.data['High'], self.data['Low'], self.data['Close'])
        # In pandas_ta, the long signal is PSARl and short is PSARs. A signal is generated when a value appears.
        self.data['PSAR_long'] = psar['PSARl_0.02_0.2']
        self.data['PSAR_short'] = psar['PSARs_0.02_0.2']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        
        # A long signal is generated when PSAR_long is not NaN, and the previous one was NaN
        buy_condition = self.data['PSAR_long'].notna() & self.data['PSAR_long'].shift(1).isna()
        # A short signal is generated when PSAR_short is not NaN, and the previous one was NaN
        sell_condition = self.data['PSAR_short'].notna() & self.data['PSAR_short'].shift(1).isna()
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy51_TsiCrossover(StrategyBase):
    """
    Strategy 51: True Strength Index (TSI) Crossover
    - Buy Signal: TSI line crosses above its signal line.
    - Sell Signal: TSI line crosses below its signal line.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        tsi = ta.tsi(self.data['Close'], fast=13, slow=25, signal=13)
        self.data['TSI'] = tsi['TSI_13_25_13']
        self.data['TSI_signal'] = tsi['TSIs_13_25_13']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['TSI'] > self.data['TSI_signal']) & (self.data['TSI'].shift(1) <= self.data['TSI_signal'].shift(1))
        sell_condition = (self.data['TSI'] < self.data['TSI_signal']) & (self.data['TSI'].shift(1) >= self.data['TSI_signal'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy52_UltimateOscillator(StrategyBase):
    """
    Strategy 52: Ultimate Oscillator
    - Buy Signal: Oscillator falls below 30 (oversold), then rises back above 35.
    - Sell Signal: Oscillator rises above 70 (overbought), then falls back below 65.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['UO'] = ta.uo(self.data['High'], self.data['Low'], self.data['Close'])
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        
        # Conditions for being in a potential buy/sell zone
        in_buy_zone = self.data['UO'] < 30
        in_sell_zone = self.data['UO'] > 70
        
        # Actual signal generation
        buy_condition = (self.data['UO'] > 35) & in_buy_zone.shift(1)
        sell_condition = (self.data['UO'] < 65) & in_sell_zone.shift(1)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy53_VortexCrossover(StrategyBase):
    """
    Strategy 53: Vortex Indicator (VI) Crossover
    - Buy Signal: Positive Vortex Indicator (VI+) crosses above Negative (VI-).
    - Sell Signal: Negative Vortex Indicator (VI-) crosses above Positive (VI+).
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        vortex = ta.vortex(self.data['High'], self.data['Low'], self.data['Close'], length=14)
        self.data['VI_plus'] = vortex['VTXP_14']
        self.data['VI_minus'] = vortex['VTXM_14']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['VI_plus'] > self.data['VI_minus']) & (self.data['VI_plus'].shift(1) <= self.data['VI_minus'].shift(1))
        sell_condition = (self.data['VI_minus'] > self.data['VI_plus']) & (self.data['VI_minus'].shift(1) >= self.data['VI_plus'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy54_FisherTransformCrossover(StrategyBase):
    """
    Strategy 54: Fisher Transform Crossover
    - Buy Signal: Fisher Transform line crosses above its signal line.
    - Sell Signal: Fisher Transform line crosses below its signal line.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        fisher = ta.fisher(self.data['High'], self.data['Low'], length=9)
        self.data['Fisher'] = fisher['FISHERt_9_1']
        self.data['Fisher_signal'] = fisher['FISHERts_9_1']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Fisher'] > self.data['Fisher_signal']) & (self.data['Fisher'].shift(1) <= self.data['Fisher_signal'].shift(1))
        sell_condition = (self.data['Fisher'] < self.data['Fisher_signal']) & (self.data['Fisher'].shift(1) >= self.data['Fisher_signal'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy55_MfiReversal(StrategyBase):
    """
    Strategy 55: Money Flow Index (MFI) Reversal
    - Buy Signal: MFI crosses above the oversold level (e.g., 20).
    - Sell Signal: MFI crosses below the overbought level (e.g., 80).
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['MFI'] = ta.mfi(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'], length=14)
        oversold_level = 20
        overbought_level = 80

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['MFI'] > oversold_level) & (self.data['MFI'].shift(1) <= oversold_level)
        sell_condition = (self.data['MFI'] < overbought_level) & (self.data['MFI'].shift(1) >= overbought_level)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy56_SqueezeMomentum(StrategyBase):
    """
    Strategy 56: Squeeze Momentum Indicator Release
    - Buy Signal: A squeeze is released (histogram becomes non-zero) and momentum is positive.
    - Sell Signal: A squeeze is released and momentum is negative.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        # pandas_ta does not have a direct squeeze_momentum, so we use ta.squeeze
        squeeze = ta.squeeze(self.data['High'], self.data['Low'], self.data['Close'])
        self.data['Squeeze_On'] = squeeze['SQZ_ON']
        self.data['Momentum'] = ta.mom(self.data['Close'], length=12) # Use standard momentum as proxy

        signals = pd.Series(0, index=self.data.index, dtype=int)
        squeeze_released = (self.data['Squeeze_On'] == 0) & (self.data['Squeeze_On'].shift(1) == 1)

        buy_condition = squeeze_released & (self.data['Momentum'] > 0)
        sell_condition = squeeze_released & (self.data['Momentum'] < 0)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy57_HmaCrossover(StrategyBase):
    """
    Strategy 57: Hull Moving Average (HMA) Crossover
    - Buy Signal: A faster HMA (e.g., 21-period) crosses above a slower HMA (e.g., 55-period).
    - Sell Signal: The faster HMA crosses below the slower HMA.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['HMA_fast'] = ta.hma(self.data['Close'], length=21)
        self.data['HMA_slow'] = ta.hma(self.data['Close'], length=55)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['HMA_fast'] > self.data['HMA_slow']) & (self.data['HMA_fast'].shift(1) <= self.data['HMA_slow'].shift(1))
        sell_condition = (self.data['HMA_fast'] < self.data['HMA_slow']) & (self.data['HMA_fast'].shift(1) >= self.data['HMA_slow'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy58_VwapBounce(StrategyBase):
    """
    Strategy 58: Long-Term VWAP Bounce (Daily Proxy)
    - Buy Signal: Price dips near or touches a long-term VWAP (e.g., 50-day) and closes higher.
    - Sell Signal: Price rises to a long-term VWAP and is rejected (closes lower).
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['VWAP_long'] = ta.vwap(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'], length=50)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Low'] < self.data['VWAP_long']) & (self.data['Close'] > self.data['VWAP_long'])
        sell_condition = (self.data['High'] > self.data['VWAP_long']) & (self.data['Close'] < self.data['VWAP_long'])

        signals[buy_condition & ~buy_condition.shift(1, fill_value=False)] = 1
        signals[sell_condition & ~sell_condition.shift(1, fill_value=False)] = -1
        return signals

class Strategy59_McGinleyDynamicCross(StrategyBase):
    """
    Strategy 59: McGinley Dynamic Line Price Crossover
    - Buy Signal: Price closes above the McGinley Dynamic Line.
    - Sell Signal: Price closes below the McGinley Dynamic Line.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        # pandas_ta doesn't have a built-in McGinley, so we implement it
        k = 0.6  # Constant, typically 60%
        length = 14
        mcginley = pd.Series(index=self.data.index, dtype=float)
        mcginley.iloc[0] = self.data['Close'].iloc[0]
        for i in range(1, len(self.data)):
            mcginley.iloc[i] = mcginley.iloc[i-1] + (self.data['Close'].iloc[i] - mcginley.iloc[i-1]) / (k * length * (self.data['Close'].iloc[i]/mcginley.iloc[i-1])**4)
        self.data['McGinley'] = mcginley

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] > self.data['McGinley']) & (self.data['Close'].shift(1) <= self.data['McGinley'].shift(1))
        sell_condition = (self.data['Close'] < self.data['McGinley']) & (self.data['Close'].shift(1) >= self.data['McGinley'].shift(1))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy60_RviCrossover(StrategyBase):
    """
    Strategy 60: Relative Vigor Index (RVI) Crossover
    - Buy Signal: RVI line crosses above its Signal line.
    - Sell Signal: RVI line crosses below its Signal line.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        rvi = ta.rvi(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'], length=14)
        self.data['RVI'] = rvi['RVI_14']
        self.data['RVI_signal'] = rvi['RVIs_14']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['RVI'] > self.data['RVI_signal']) & (self.data['RVI'].shift(1) <= self.data['RVI_signal'].shift(1))
        sell_condition = (self.data['RVI'] < self.data['RVI_signal']) & (self.data['RVI'].shift(1) >= self.data['RVI_signal'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals
        
class Strategy61_AtrBreakout(StrategyBase):
    """
    Strategy 61: Average True Range (ATR) Volatility Breakout
    - Buy Signal: Price closes above a moving average plus a multiple of ATR.
    - Sell Signal: Price closes below a moving average minus a multiple of ATR.
    """
    def __init__(self, data):
        super().__init__(data)
        
    def generate_signals(self):
        self.data['SMA_20'] = ta.sma(self.data['Close'], length=20)
        self.data['ATR'] = ta.atr(self.data['High'], self.data['Low'], self.data['Close'], length=14)
        atr_multiplier = 2.0

        upper_band = self.data['SMA_20'] + (self.data['ATR'] * atr_multiplier)
        lower_band = self.data['SMA_20'] - (self.data['ATR'] * atr_multiplier)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] > upper_band) & (self.data['Close'].shift(1) <= upper_band.shift(1))
        sell_condition = (self.data['Close'] < lower_band) & (self.data['Close'].shift(1) >= lower_band.shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy62_KstCrossover(StrategyBase):
    """
    Strategy 62: Know Sure Thing (KST) Crossover
    - Buy Signal: The KST line crosses above its signal line.
    - Sell Signal: The KST line crosses below its signal line.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        kst = ta.kst(self.data['Close'])
        self.data['KST'] = kst['KST_10_15_20_30_10_10_10_15']
        self.data['KST_signal'] = kst['KSTs_9']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['KST'] > self.data['KST_signal']) & (self.data['KST'].shift(1) <= self.data['KST_signal'].shift(1))
        sell_condition = (self.data['KST'] < self.data['KST_signal']) & (self.data['KST'].shift(1) >= self.data['KST_signal'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy63_BullBearPower(StrategyBase):
    """
    Strategy 63: Elder's Bull and Bear Power
    - Buy Signal: Bull Power crosses above zero while EMA is rising.
    - Sell Signal: Bear Power crosses below zero while EMA is falling.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        ema = ta.ema(self.data['Close'], length=13)
        self.data['Bull_Power'] = self.data['High'] - ema
        self.data['Bear_Power'] = self.data['Low'] - ema

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Bull_Power'] > 0) & (self.data['Bull_Power'].shift(1) <= 0) & (ema > ema.shift(1))
        sell_condition = (self.data['Bear_Power'] < 0) & (self.data['Bear_Power'].shift(1) >= 0) & (ema < ema.shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy64_EngulfingCandle(StrategyBase):
    """
    Strategy 64: Bullish and Bearish Engulfing Candlestick Patterns
    - Buy Signal: A Bullish Engulfing pattern is detected.
    - Sell Signal: A Bearish Engulfing pattern is detected.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        engulfing = ta.cdl_pattern(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'], name="engulfing")
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        # Bullish Engulfing returns 100, Bearish returns -100
        signals[engulfing == 100] = 1
        signals[engulfing == -100] = -1
        return signals

class Strategy65_HammerHangingMan(StrategyBase):
    """
    Strategy 65: Hammer and Hanging Man Candlestick Patterns with Trend Filter
    - Buy Signal: A Hammer pattern appears in a downtrend.
    - Sell Signal: A Hanging Man pattern appears in an uptrend.
    """

    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['SMA_50'] = ta.sma(self.data['Close'], length=50)
        hammer = ta.cdl_pattern(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'], name="hammer")
        hanging_man = ta.cdl_pattern(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'],
                                     name="hangingman")

        downtrend = self.data['Close'] < self.data['SMA_50']
        uptrend = self.data['Close'] > self.data['SMA_50']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        signals[(hammer != 0) & downtrend] = 1
        signals[(hanging_man != 0) & uptrend] = -1
        return signals

class Strategy66_MorningEveningStar(StrategyBase):
    """
    Strategy 66: Morning Star and Evening Star Candlestick Patterns
    - Buy Signal: A Morning Star pattern is detected (bullish reversal).
    - Sell Signal: An Evening Star pattern is detected (bearish reversal).
    """
    def __init__(self, data):
        super().__init__(data)
        
    def generate_signals(self):
        morning_star = ta.cdl_pattern(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'], name="morningstar")
        evening_star = ta.cdl_pattern(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'], name="eveningstar")
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        signals[morning_star != 0] = 1
        signals[evening_star != 0] = -1
        return signals

class Strategy67_GuppyMma(StrategyBase):
    """
    Strategy 67: Guppy Multiple Moving Averages (GMMA) Crossover
    - Buy Signal: The group of short-term EMAs crosses above the group of long-term EMAs.
    - Sell Signal: The group of short-term EMAs crosses below the group of long-term EMAs.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        gmma = ta.gmma(self.data['Close'])
        short_term_avg = gmma[['GMMA_3', 'GMMA_5', 'GMMA_8', 'GMMA_10', 'GMMA_12', 'GMMA_15']].mean(axis=1)
        long_term_avg = gmma[['GMMA_30', 'GMMA_35', 'GMMA_40', 'GMMA_45', 'GMMA_50', 'GMMA_60']].mean(axis=1)
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (short_term_avg > long_term_avg) & (short_term_avg.shift(1) <= long_term_avg.shift(1))
        sell_condition = (short_term_avg < long_term_avg) & (short_term_avg.shift(1) >= long_term_avg.shift(1))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy68_DonchianVolume(StrategyBase):
    """
    Strategy 68: Donchian Channel Breakout with Volume Confirmation
    - Buy Signal: Close breaks above the upper Donchian channel with volume above its 20-day average.
    - Sell Signal: Close breaks below the lower Donchian channel with volume above its 20-day average.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        donchian = ta.donchian(self.data['High'], self.data['Low'], lower_length=20, upper_length=20)
        self.data['DC_upper'] = donchian['DCU_20_20']
        self.data['DC_lower'] = donchian['DCL_20_20']
        self.data['Volume_SMA'] = ta.sma(self.data['Volume'], length=20)
        
        high_volume = self.data['Volume'] > self.data['Volume_SMA']
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] > self.data['DC_upper'].shift(1)) & high_volume
        sell_condition = (self.data['Close'] < self.data['DC_lower'].shift(1)) & high_volume

        signals[buy_condition & ~buy_condition.shift(1, fill_value=False)] = 1
        signals[sell_condition & ~sell_condition.shift(1, fill_value=False)] = -1
        return signals

class Strategy69_CenterOfGravity(StrategyBase):
    """
    Strategy 69: Center of Gravity (COG) Oscillator Crossover
    - Buy Signal: COG line crosses above its signal line.
    - Sell Signal: COG line crosses below its signal line.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        # pandas_ta doesn't have COG, so it's calculated manually
        length = 10
        prices = self.data['Close']
        numerator = pd.Series(index=prices.index, dtype=float)
        denominator = prices.rolling(window=length).sum()
        for i in range(len(prices)):
            if i >= length -1:
                num = 0
                for j in range(length):
                    num += (j + 1) * prices.iloc[i - j]
                numerator.iloc[i] = -num
        
        self.data['COG'] = numerator / denominator
        self.data['COG_signal'] = ta.sma(self.data['COG'], length=3) # Simple SMA as signal line

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['COG'] > self.data['COG_signal']) & (self.data['COG'].shift(1) <= self.data['COG_signal'].shift(1))
        sell_condition = (self.data['COG'] < self.data['COG_signal']) & (self.data['COG'].shift(1) >= self.data['COG_signal'].shift(1))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy70_WoodiesCci(StrategyBase):
    """
    Strategy 70: Woodie's CCI Breakout
    - Buy Signal: CCI crosses above 100.
    - Sell Signal: CCI crosses below -100.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['CCI'] = ta.cci(self.data['High'], self.data['Low'], self.data['Close'], length=14)
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['CCI'] > 100) & (self.data['CCI'].shift(1) <= 100)
        sell_condition = (self.data['CCI'] < -100) & (self.data['CCI'].shift(1) >= -100)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals
        
class Strategy71_TripleSuperTrend(StrategyBase):
    """
    Strategy 71: Triple SuperTrend Confirmation
    - Buy Signal: All three SuperTrend indicators (with different multipliers) flip to an uptrend.
    - Sell Signal: All three SuperTrend indicators flip to a downtrend.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        st1 = ta.supertrend(self.data['High'], self.data['Low'], self.data['Close'], length=10, multiplier=1.0)
        st2 = ta.supertrend(self.data['High'], self.data['Low'], self.data['Close'], length=11, multiplier=2.0)
        st3 = ta.supertrend(self.data['High'], self.data['Low'], self.data['Close'], length=12, multiplier=3.0)
        
        # Dynamically get the direction column which is the first column
        st1_col = st1.columns[0]
        st2_col = st2.columns[0]
        st3_col = st3.columns[0]

        self.data['ST_dir1'] = st1[st1_col]
        self.data['ST_dir2'] = st2[st2_col]
        self.data['ST_dir3'] = st3[st3_col]

        all_up = (self.data['ST_dir1'] == 1) & (self.data['ST_dir2'] == 1) & (self.data['ST_dir3'] == 1)
        all_down = (self.data['ST_dir1'] == -1) & (self.data['ST_dir2'] == -1) & (self.data['ST_dir3'] == -1)
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        signals[all_up & ~all_up.shift(1, fill_value=False)] = 1
        signals[all_down & ~all_down.shift(1, fill_value=False)] = -1
        return signals

class Strategy72_DpoCrossover(StrategyBase):
    """
    Strategy 72: Detrended Price Oscillator (DPO) Zero Crossover
    - Buy Signal: DPO crosses above zero.
    - Sell Signal: DPO crosses below zero.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['DPO'] = ta.dpo(self.data['Close'], length=20)
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['DPO'] > 0) & (self.data['DPO'].shift(1) <= 0)
        sell_condition = (self.data['DPO'] < 0) & (self.data['DPO'].shift(1) >= 0)
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy73_QqeCrossover(StrategyBase):
    """
    Strategy 73: Quantitative Qualitative Estimation (QQE) Crossover
    - Buy Signal: The QQE line crosses above the signal line.
    - Sell Signal: The QQE line crosses below the signal line.
    """
    def __init__(self, data):
        super().__init__(data)
        
    def generate_signals(self):
        qqe = ta.qqe(self.data['Close'], length=14)
        
        # Dynamically get column names
        qqe_line_col = qqe.columns[0]
        qqe_signal_col = qqe.columns[1]

        self.data['QQE_line'] = qqe[qqe_line_col]
        self.data['QQE_signal'] = qqe[qqe_signal_col]
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['QQE_line'] > self.data['QQE_signal']) & (self.data['QQE_line'].shift(1) <= self.data['QQE_signal'].shift(1))
        sell_condition = (self.data['QQE_line'] < self.data['QQE_signal']) & (self.data['QQE_line'].shift(1) >= self.data['QQE_signal'].shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy74_IchimokuKijunSenCross(StrategyBase):
    """
    Strategy 74: Ichimoku Kijun-Sen Cross Strategy
    - Buy Signal: Price crosses above the Kijun-Sen (Base Line).
    - Sell Signal: Price crosses below the Kijun-Sen.
    """
    def __init__(self, data, tenkan=9, kijun=26, senkou=52):
        super().__init__(data)
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou = senkou

    def generate_signals(self):
        ichimoku_df, _ = ta.ichimoku(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            tenkan=self.tenkan,
            kijun=self.kijun,
            senkou=self.senkou
        )

        self.data['Kijun_Sen'] = ichimoku_df[f'IKS_{self.kijun}']

        signals = pd.Series(0, index=self.data.index, dtype=int)

        price_cross_above = (
            (self.data['Close'] > self.data['Kijun_Sen']) &
            (self.data['Close'].shift(1) <= self.data['Kijun_Sen'].shift(1))
        )
        price_cross_below = (
            (self.data['Close'] < self.data['Kijun_Sen']) &
            (self.data['Close'].shift(1) >= self.data['Kijun_Sen'].shift(1))
        )

        signals[price_cross_above] = 1
        signals[price_cross_below] = -1

        return signals


class Strategy75_SmiCrossover(StrategyBase):
    """
    Strategy 75: Stochastic Momentum Index (SMI) Crossover
    - Buy Signal: SMI line crosses above its signal line.
    - Sell Signal: SMI line crosses below its signal line.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        smi = ta.smi(self.data['Close'])
        self.data['SMI_line'] = smi['SMI_5_20_5']
        self.data['SMI_signal'] = smi['SMIs_5_20_5']
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['SMI_line'] > self.data['SMI_signal']) & (self.data['SMI_line'].shift(1) <= self.data['SMI_signal'].shift(1))
        sell_condition = (self.data['SMI_line'] < self.data['SMI_signal']) & (self.data['SMI_line'].shift(1) >= self.data['SMI_signal'].shift(1))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy76_BalanceOfPower(StrategyBase):
    """
    Strategy 76: Balance of Power (BOP) Zero Crossover
    - Buy Signal: BOP indicator crosses above zero.
    - Sell Signal: BOP indicator crosses below zero.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['BOP'] = ta.bop(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['BOP'] > 0) & (self.data['BOP'].shift(1) <= 0)
        sell_condition = (self.data['BOP'] < 0) & (self.data['BOP'].shift(1) >= 0)
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy77_EmaRibbonExpansion(StrategyBase):
    """
    Strategy 77: EMA Ribbon Expansion
    - Buy Signal: EMA ribbon, after a period of compression, expands with short EMAs above long EMAs.
    - Sell Signal: Ribbon expands with short EMAs below long EMAs.
    """
    def __init__(self, data):
        super().__init__(data)
        
    def generate_signals(self):
        self.data['EMA8'] = ta.ema(self.data['Close'], length=8)
        self.data['EMA13'] = ta.ema(self.data['Close'], length=13)
        self.data['EMA21'] = ta.ema(self.data['Close'], length=21)
        self.data['EMA55'] = ta.ema(self.data['Close'], length=55)
        
        ribbon_width = (self.data['EMA8'] - self.data['EMA55']).abs()
        ribbon_width_sma = ta.sma(ribbon_width, length=10)
        
        is_squeezed = ribbon_width < ribbon_width_sma
        is_expanding = ribbon_width > ribbon_width_sma
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        
        buy_condition = is_expanding & is_squeezed.shift(1) & (self.data['EMA8'] > self.data['EMA55'])
        sell_condition = is_expanding & is_squeezed.shift(1) & (self.data['EMA8'] < self.data['EMA55'])
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy78_ZScoreReversion(StrategyBase):
    """
    Strategy 78: Z-Score Mean Reversion
    - Buy Signal: The price's Z-Score falls below a negative threshold (e.g., -2.0).
    - Sell Signal: The price's Z-Score rises above a positive threshold (e.g., 2.0).
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        length = 20
        price_mean = self.data['Close'].rolling(window=length).mean()
        price_std = self.data['Close'].rolling(window=length).std()
        self.data['Z_Score'] = (self.data['Close'] - price_mean) / price_std
        
        buy_threshold = -2.0
        sell_threshold = 2.0
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Z_Score'] < buy_threshold) & (self.data['Z_Score'].shift(1) >= buy_threshold)
        sell_condition = (self.data['Z_Score'] > sell_threshold) & (self.data['Z_Score'].shift(1) <= sell_threshold)
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy79_VolumeSpikeReversal(StrategyBase):
    """
    Strategy 79: Volume Spike Contrarian Reversal
    - Buy Signal: A red candle appears on a volume spike (e.g., > 3x avg volume). Signal is for next day.
    - Sell Signal: A green candle appears on a volume spike. Signal is for next day.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['Volume_SMA'] = ta.sma(self.data['Volume'], length=20)
        volume_spike = self.data['Volume'] > 3 * self.data['Volume_SMA']
        
        is_green_candle = self.data['Close'] > self.data['Open']
        is_red_candle = self.data['Close'] < self.data['Open']
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        
        # Buy signal on the day AFTER a red candle volume spike
        signals[(volume_spike & is_red_candle).shift(1, fill_value=False)] = 1
        # Sell signal on the day AFTER a green candle volume spike
        signals[(volume_spike & is_green_candle).shift(1, fill_value=False)] = -1
        
        return signals

class Strategy80_AwesomeMacdCombo(StrategyBase):
    """
    Strategy 80: Awesome Oscillator and MACD Combo
    - Buy Signal: Awesome Oscillator crosses above zero AND the MACD line is above its signal line.
    - Sell Signal: Awesome Oscillator crosses below zero AND the MACD line is below its signal line.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['AO'] = ta.ao(self.data['High'], self.data['Low'])
        macd = ta.macd(self.data['Close'])
        self.data['MACD'] = macd['MACD_12_26_9']
        self.data['MACD_signal'] = macd['MACDs_12_26_9']
        
        ao_cross_up = (self.data['AO'] > 0) & (self.data['AO'].shift(1) <= 0)
        ao_cross_down = (self.data['AO'] < 0) & (self.data['AO'].shift(1) >= 0)
        
        macd_bullish = self.data['MACD'] > self.data['MACD_signal']
        macd_bearish = self.data['MACD'] < self.data['MACD_signal']
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        signals[ao_cross_up & macd_bullish] = 1
        signals[ao_cross_down & macd_bearish] = -1
        return signals

class Strategy81_TtmSqueezePro(StrategyBase):
    """
    Strategy 81: TTM Squeeze Pro
    - Buy Signal: A squeeze is released (squeeze_off) and the histogram is positive and rising.
    - Sell Signal: A squeeze is released (squeeze_off) and the histogram is negative and falling.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        squeeze_pro = ta.squeeze_pro(self.data['High'], self.data['Low'], self.data['Close'])
        
        # Dynamically find the correct column names
        sqz_on_col = [col for col in squeeze_pro.columns if 'SQZPRO_ON' in col][0]
        sqz_hist_col = [col for col in squeeze_pro.columns if 'SQZPRO' in col and 'ON' not in col and 'OFF' not in col and 'NO' not in col][0]
        
        self.data['SQZPRO_ON'] = squeeze_pro[sqz_on_col]
        self.data['SQZPRO_HIST'] = squeeze_pro[sqz_hist_col]
        
        squeeze_released = (self.data['SQZPRO_ON'] == 0) & (self.data['SQZPRO_ON'].shift(1) == 1)
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = squeeze_released & (self.data['SQZPRO_HIST'] > 0) & (self.data['SQZPRO_HIST'] > self.data['SQZPRO_HIST'].shift(1))
        sell_condition = squeeze_released & (self.data['SQZPRO_HIST'] < 0) & (self.data['SQZPRO_HIST'] < self.data['SQZPRO_HIST'].shift(1))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy82_DojiReversal(StrategyBase):
    """
    Strategy 82: Doji Candlestick Reversal with Trend Filter
    - Buy Signal: A Doji appears in a clear downtrend.
    - Sell Signal: A Doji appears in a clear uptrend.
    """

    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['SMA_30'] = ta.sma(self.data['Close'], length=30)
        doji = ta.cdl_pattern(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'], name="doji")

        in_downtrend = self.data['Close'] < self.data['SMA_30']
        in_uptrend = self.data['Close'] > self.data['SMA_30']

        signals = pd.Series(0, index=self.data.index, dtype=int)
        signals[(doji != 0) & in_downtrend] = 1
        signals[(doji != 0) & in_uptrend] = -1
        return signals

class Strategy83_AroonClassicCrossover(StrategyBase):
    """
    Strategy 83: Aroon Up/Down Crossover
    - Buy Signal: Aroon Up crosses above Aroon Down.
    - Sell Signal: Aroon Down crosses above Aroon Up.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        aroon = ta.aroon(self.data['High'], self.data['Low'], length=25)
        self.data['Aroon_Up'] = aroon['AROONu_25']
        self.data['Aroon_Down'] = aroon['AROONd_25']
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Aroon_Up'] > self.data['Aroon_Down']) & (self.data['Aroon_Up'].shift(1) <= self.data['Aroon_Down'].shift(1))
        sell_condition = (self.data['Aroon_Down'] > self.data['Aroon_Up']) & (self.data['Aroon_Down'].shift(1) >= self.data['Aroon_Up'].shift(1))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy84_PsychologicalLine(StrategyBase):
    """
    Strategy 84: Psychological Line (PSY) Reversal
    - Buy Signal: PSY crosses below an oversold level (e.g., 25).
    - Sell Signal: PSY crosses above an overbought level (e.g., 75).
    """
    def __init__(self, data):
        super().__init__(data)
        
    def generate_signals(self):
        length = 12
        up_days = (self.data['Close'] > self.data['Close'].shift(1)).rolling(window=length).sum()
        self.data['PSY'] = (up_days / length) * 100
        
        oversold = 25
        overbought = 75
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['PSY'] < oversold) & (self.data['PSY'].shift(1) >= oversold)
        sell_condition = (self.data['PSY'] > overbought) & (self.data['PSY'].shift(1) <= overbought)
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy85_VwapDeviation(StrategyBase):
    """
    Strategy 85: Volume Weighted Average Price (VWAP) Deviation
    - Buy Signal: Close crosses below the lower VWAP band (VWAP - N * StdDev).
    - Sell Signal: Close crosses above the upper VWAP band (VWAP + N * StdDev).
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        length = 20
        stdev_multiplier = 2.0
        self.data['VWAP'] = ta.vwap(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'], length=length)
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        self.data['VWAP_StDev'] = typical_price.rolling(window=length).std()
        
        upper_band = self.data['VWAP'] + (self.data['VWAP_StDev'] * stdev_multiplier)
        lower_band = self.data['VWAP'] - (self.data['VWAP_StDev'] * stdev_multiplier)
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] < lower_band)
        sell_condition = (self.data['Close'] > upper_band)
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy86_SchaffTrendCycle(StrategyBase):
    """
    Strategy 86: Schaff Trend Cycle (STC)
    - Buy Signal: STC crosses up from the oversold zone (e.g., below 25).
    - Sell Signal: STC crosses down from the overbought zone (e.g., above 75).
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        stc_df = ta.stc(self.data['Close'])
        
        # Dynamically get the STC column name (first column)
        stc_col = stc_df.columns[0]
        self.data['STC'] = stc_df[stc_col]
        
        oversold = 25
        overbought = 75
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['STC'] > oversold) & (self.data['STC'].shift(1) <= oversold)
        sell_condition = (self.data['STC'] < overbought) & (self.data['STC'].shift(1) >= overbought)
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy87_LinRegSlope(StrategyBase):
    """
    Strategy 87: Linear Regression Slope Change
    - Buy Signal: The slope of the linear regression line turns from negative to positive.
    - Sell Signal: The slope turns from positive to negative.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['LinReg_Slope'] = ta.linreg(self.data['Close'], length=14, slope=True)['LRS_14']
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['LinReg_Slope'] > 0) & (self.data['LinReg_Slope'].shift(1) <= 0)
        sell_condition = (self.data['LinReg_Slope'] < 0) & (self.data['LinReg_Slope'].shift(1) >= 0)
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy88_TrixZeroCross(StrategyBase):
    """
    Strategy 88: Triple EMA (TRIX) Zero Line Crossover
    - Buy Signal: TRIX indicator crosses above zero.
    - Sell Signal: TRIX indicator crosses below zero.
    """
    def __init__(self, data):
        super().__init__(data)
        
    def generate_signals(self):
        self.data['TRIX'] = ta.trix(self.data['Close'], length=15)['TRIX_15_9']
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['TRIX'] > 0) & (self.data['TRIX'].shift(1) <= 0)
        sell_condition = (self.data['TRIX'] < 0) & (self.data['TRIX'].shift(1) >= 0)

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy89_ThreeSoldiersCrows(StrategyBase):
    """
    Strategy 89: Three White Soldiers / Three Black Crows
    - Buy Signal: Three White Soldiers pattern is detected.
    - Sell Signal: Three Black Crows pattern is detected.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        three_soldiers = ta.cdl_pattern(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'], name="3whitesoldiers")
        three_crows = ta.cdl_pattern(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'], name="3blackcrows")
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        signals[three_soldiers != 0] = 1
        signals[three_crows != 0] = -1
        return signals

class Strategy90_HmaDirection(StrategyBase):
    """
    Strategy 90: Hull Moving Average (HMA) Direction Change
    - Buy Signal: The HMA (21-period) changes direction from down to up.
    - Sell Signal: The HMA changes direction from up to down.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['HMA'] = ta.hma(self.data['Close'], length=21)
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['HMA'] > self.data['HMA'].shift(1)) & (self.data['HMA'].shift(1) <= self.data['HMA'].shift(2))
        sell_condition = (self.data['HMA'] < self.data['HMA'].shift(1)) & (self.data['HMA'].shift(1) >= self.data['HMA'].shift(2))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy91_ChandeKrollStopCross(StrategyBase):
    """
    Strategy 91: Chande Kroll Stop Cross (used for entry signals)
    - Buy Signal: Price crosses above the long stop line.
    - Sell Signal: Price crosses below the short stop line.
    Note: The error for this strategy was not visible in the provided image.
    The code has been reviewed for robustness.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        # Parameters for Chande Kroll Stop
        atr_period = 10
        atr_multiplier = 1.0
        stop_period = 9
        
        # Calculate ATR
        atr = ta.atr(self.data['High'], self.data['Low'], self.data['Close'], length=atr_period)
        
        # Calculate Stops
        long_stop = self.data['High'].rolling(window=stop_period).max() - (atr * atr_multiplier)
        short_stop = self.data['Low'].rolling(window=stop_period).min() + (atr * atr_multiplier)

        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] > long_stop) & (self.data['Close'].shift(1) <= long_stop.shift(1))
        sell_condition = (self.data['Close'] < short_stop) & (self.data['Close'].shift(1) >= short_stop.shift(1))

        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals
        
class Strategy92_PriceRibbonCross(StrategyBase):
    """
    Strategy 92: Price Crossing an EMA Ribbon
    - Buy Signal: Price crosses above the entire EMA ribbon (fastest and slowest EMAs).
    - Sell Signal: Price crosses below the entire EMA ribbon.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['EMA_fast'] = ta.ema(self.data['Close'], length=10)
        self.data['EMA_slow'] = ta.ema(self.data['Close'], length=50)
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] > self.data['EMA_fast']) & (self.data['Close'] > self.data['EMA_slow']) & \
                        ((self.data['Close'].shift(1) <= self.data['EMA_fast'].shift(1)) | (self.data['Close'].shift(1) <= self.data['EMA_slow'].shift(1)))
                        
        sell_condition = (self.data['Close'] < self.data['EMA_fast']) & (self.data['Close'] < self.data['EMA_slow']) & \
                         ((self.data['Close'].shift(1) >= self.data['EMA_fast'].shift(1)) | (self.data['Close'].shift(1) >= self.data['EMA_slow'].shift(1)))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy93_VolatilityPercentile(StrategyBase):
    """
    Strategy 93: Historical Volatility Percentile Breakout
    - Buy Signal: Volatility drops to a low percentile (e.g., <10%), anticipating a breakout.
    - Sell Signal: Volatility rises to a high percentile (e.g., >90%), anticipating mean reversion/calm.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['Volatility'] = log_returns.rolling(window=21).std() * np.sqrt(252)
        self.data['Vol_Percentile'] = self.data['Volatility'].rolling(window=100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        low_percentile = 0.10
        high_percentile = 0.90
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Vol_Percentile'] < low_percentile) & (self.data['Vol_Percentile'].shift(1) >= low_percentile)
        sell_condition = (self.data['Vol_Percentile'] > high_percentile) & (self.data['Vol_Percentile'].shift(1) <= high_percentile)
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy94_CoppockTrendFilter(StrategyBase):
    """
    Strategy 94: Coppock Curve with a Long-Term Trend Filter
    - Buy Signal: Coppock curve crosses above zero and price is above the 200-day SMA.
    - Sell Signal: Coppock curve crosses below zero and price is below the 200-day SMA.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['Coppock'] = ta.coppock(self.data['Close'])
        self.data['SMA_200'] = ta.sma(self.data['Close'], length=200)
        
        coppock_cross_up = (self.data['Coppock'] > 0) & (self.data['Coppock'].shift(1) <= 0)
        coppock_cross_down = (self.data['Coppock'] < 0) & (self.data['Coppock'].shift(1) >= 0)
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        signals[coppock_cross_up & (self.data['Close'] > self.data['SMA_200'])] = 1
        signals[coppock_cross_down & (self.data['Close'] < self.data['SMA_200'])] = -1
        return signals

class Strategy95_EldersImpulse(StrategyBase):
    """
    Strategy 95: Elder's Impulse System
    - Buy Signal: A 'green' bar occurs (EMA rising and MACD-Hist rising).
    - Sell Signal: A 'red' bar occurs (EMA falling and MACD-Hist falling).
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['EMA_13'] = ta.ema(self.data['Close'], length=13)
        macd = ta.macd(self.data['Close'])
        self.data['MACD_Hist'] = macd['MACDh_12_26_9']
        
        ema_rising = self.data['EMA_13'] > self.data['EMA_13'].shift(1)
        hist_rising = self.data['MACD_Hist'] > self.data['MACD_Hist'].shift(1)
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = ema_rising & hist_rising
        sell_condition = ~ema_rising & ~hist_rising
        
        signals[buy_condition & ~buy_condition.shift(1, fill_value=False)] = 1
        signals[sell_condition & ~sell_condition.shift(1, fill_value=False)] = -1
        return signals
        
class Strategy96_DonchianMidline(StrategyBase):
    """
    Strategy 96: Donchian Channel Midline Reversion
    - Buy Signal: Price crosses above the midline of the Donchian Channel.
    - Sell Signal: Price crosses below the midline of the Donchian Channel.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        donchian = ta.donchian(self.data['High'], self.data['Low'], length=20)
        self.data['DCM'] = donchian['DCM_20'] # Midline
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        buy_condition = (self.data['Close'] > self.data['DCM']) & (self.data['Close'].shift(1) <= self.data['DCM'].shift(1))
        sell_condition = (self.data['Close'] < self.data['DCM']) & (self.data['Close'].shift(1) >= self.data['DCM'].shift(1))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals

class Strategy97_GatorOscillator(StrategyBase):
    """
    Strategy 97: Gator Oscillator State Change
    - Buy Signal: Gator "wakes up" and starts "eating" (first green bar after blue).
    - Sell Signal: Gator is "sated" (first red bar after a green sequence).
    Note: The error for this strategy was not visible in the provided image. 
    The code has been updated to be more robust against parameter changes.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        # Uses Alligator jaws, teeth, lips
        alligator = ta.alligator(self.data['High'], self.data['Low'])

        # Dynamically get column names to make it robust
        jaw_col = [col for col in alligator.columns if 'JAW' in col][0]
        teeth_col = [col for col in alligator.columns if 'TEETH' in col][0]
        lips_col = [col for col in alligator.columns if 'LIPS' in col][0]

        gator_upper = (alligator[jaw_col] - alligator[teeth_col]).abs()
        gator_lower = -(alligator[teeth_col] - alligator[lips_col]).abs()

        is_green = gator_upper > gator_upper.shift(1)
        is_blue = gator_upper < gator_upper.shift(1)
        is_red = (gator_upper < gator_upper.shift(1)) & (gator_lower > gator_lower.shift(1))

        signals = pd.Series(0, index=self.data.index, dtype=int)
        # Buy when it wakes up (green bar appears after blue)
        signals[is_green & is_blue.shift(1)] = 1
        # Sell when sated (red bar appears after green)
        signals[is_red & is_green.shift(1)] = -1
        return signals

class Strategy98_RocVolumeConfirm(StrategyBase):
    """
    Strategy 98: Rate of Change (ROC) with Volume Confirmation
    - Buy Signal: Price ROC crosses above zero and Volume ROC is also positive.
    - Sell Signal: Price ROC crosses below zero and Volume ROC is also positive.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['ROC_Price'] = ta.roc(self.data['Close'], length=12)
        self.data['ROC_Volume'] = ta.roc(self.data['Volume'], length=12)

        price_cross_up = (self.data['ROC_Price'] > 0) & (self.data['ROC_Price'].shift(1) <= 0)
        price_cross_down = (self.data['ROC_Price'] < 0) & (self.data['ROC_Price'].shift(1) >= 0)
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        signals[price_cross_up & (self.data['ROC_Volume'] > 0)] = 1
        signals[price_cross_down & (self.data['ROC_Volume'] > 0)] = -1
        return signals

class Strategy99_MarketFacilitation(StrategyBase):
    """
    Strategy 99: Market Facilitation Index (MFI_BW)
    - Buy Signal: A "Green" bar appears (MFI and Volume are both up).
    - Sell Signal: A "Squat" bar appears (MFI up, Volume down), suggesting a potential reversal.
    """

    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        self.data['MFI_BW'] = (self.data['High'] - self.data['Low']) / self.data['Volume']

        mfi_up = self.data['MFI_BW'] > self.data['MFI_BW'].shift(1)
        vol_up = self.data['Volume'] > self.data['Volume'].shift(1)

        green_bar = mfi_up & vol_up
        squat_bar = mfi_up & ~vol_up

        signals = pd.Series(0, index=self.data.index, dtype=int)
        signals[green_bar & ~green_bar.shift(1, fill_value=False)] = 1
        signals[squat_bar & ~squat_bar.shift(1, fill_value=False)] = -1
        return signals

class Strategy100_HeikinAshiSupertrend(StrategyBase):
    """
    Strategy 100: Heikin Ashi Signal with Supertrend Filter
    - Buy Signal: Heikin Ashi turns green while price is above the Supertrend line.
    - Sell Signal: Heikin Ashi turns red while price is below the Supertrend line.
    """
    def __init__(self, data):
        super().__init__(data)

    def generate_signals(self):
        ha_data = ta.ha(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close'])
        supertrend = ta.supertrend(self.data['High'], self.data['Low'], self.data['Close'])['SUPERTd_7_3.0']

        ha_green = ha_data['HA_close'] > ha_data['HA_open']
        ha_red = ha_data['HA_close'] < ha_data['HA_open']
        
        above_supertrend = supertrend == 1
        below_supertrend = supertrend == -1
        
        signals = pd.Series(0, index=self.data.index, dtype=int)
        signals[ha_green & ~ha_green.shift(1, fill_value=False) & above_supertrend] = 1
        signals[ha_red & ~ha_red.shift(1, fill_value=False) & below_supertrend] = -1
        return signals