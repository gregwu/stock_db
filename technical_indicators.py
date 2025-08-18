#!/usr/bin/env python3
"""
technical_indicators.py - Shared Technical Indicators Library

This module provides a comprehensive collection of technical indicator calculations
that can be shared across different modules like scanner.py and fractal_predict.py.

All functions expect pandas Series or DataFrame columns as input and return
the calculated indicator values.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Union


class TechnicalIndicators:
    """Collection of technical indicator calculation methods"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            data: Price series (typically close price)
            period: RSI calculation period (default 14)
            
        Returns:
            RSI values
        """
        delta = data.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # Calculate the first average gain and loss using simple moving average
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Calculate subsequent values using Wilder's smoothing
        for i in range(period, len(avg_gain)):
            if not pd.isna(gain.iloc[i]):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series (typically close price)
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Args:
            data: Price series (typically close price)
            window: Moving average window (default 20)
            std_dev: Standard deviation multiplier (default 2)
            
        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        middle_band = data.rolling(window=window).mean()
        std = data.rolling(window=window).std(ddof=0)  # Population std for BB
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return middle_band, upper_band, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series  
            close: Close price series
            k_period: %K calculation period (default 14)
            d_period: %D smoothing period (default 3)
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        d_percent = k_percent.rolling(d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV values
        """
        price_change = close.diff()
        obv_values = []
        obv = 0
        
        for i in range(len(close)):
            if i == 0 or pd.isna(price_change.iloc[i]):
                obv_values.append(obv)
            elif price_change.iloc[i] > 0:
                obv += volume.iloc[i]
                obv_values.append(obv)
            elif price_change.iloc[i] < 0:
                obv -= volume.iloc[i]
                obv_values.append(obv)
            else:  # price_change == 0
                obv_values.append(obv)
        
        return pd.Series(obv_values, index=close.index)
    
    @staticmethod
    def price_distance_to_ma(price: pd.Series, ma_length: int = 20, signal_length: int = 9, exponential: bool = False) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Price Distance to Moving Average indicator
        
        Args:
            price: Price series (typically close price)
            ma_length: Moving average length (default 20)
            signal_length: Signal line length (default 9)
            exponential: Use EMA instead of SMA (default False)
            
        Returns:
            Tuple of (pma, signal, cycle)
        """
        # Calculate moving average
        if exponential:
            ma = TechnicalIndicators.ema(price, ma_length)
            signal_ma_func = TechnicalIndicators.ema
        else:
            ma = TechnicalIndicators.sma(price, ma_length)
            signal_ma_func = TechnicalIndicators.sma
        
        # Calculate price distance to MA as percentage
        pma = ((price / ma) - 1) * 100
        
        # Calculate signal line (MA of PMA)
        signal = signal_ma_func(pma, signal_length)
        
        # Calculate cycle (difference between PMA and signal)
        cycle = pma - signal
        
        return pma, signal, cycle
    
    @staticmethod
    def pma_threshold_bands(pma: pd.Series, bb_length: int = 200, std_dev_low: float = 1.5, std_dev_high: float = 2.25) -> Dict[str, pd.Series]:
        """
        Calculate threshold bands for PMA using Bollinger Bands methodology
        
        Args:
            pma: Price/MA ratio series
            bb_length: Bollinger Bands calculation length
            std_dev_low: Lower standard deviation multiplier
            std_dev_high: Higher standard deviation multiplier
            
        Returns:
            Dictionary with upper and lower threshold bands
        """
        # Calculate Bollinger Bands components
        bb_sma = pma.rolling(window=bb_length).mean()
        bb_std = pma.rolling(window=bb_length).std()
        
        # Calculate threshold bands
        upper_low = bb_sma + (bb_std * std_dev_low)
        lower_low = bb_sma - (bb_std * std_dev_low)
        upper_high = bb_sma + (bb_std * std_dev_high)
        lower_high = bb_sma - (bb_std * std_dev_high)
        
        return {
            'upper_low': upper_low,
            'lower_low': lower_low,
            'upper_high': upper_high,
            'lower_high': lower_high
        }
    
    @staticmethod
    def volatility(price_change: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate volatility (annualized standard deviation)
        
        Args:
            price_change: Price change series (pct_change)
            window: Rolling window for calculation
            
        Returns:
            Annualized volatility
        """
        return price_change.rolling(window).std() * np.sqrt(252)
    
    @staticmethod
    def support_resistance(high: pd.Series, low: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate support and resistance levels
        
        Args:
            high: High price series
            low: Low price series
            window: Rolling window for calculation
            
        Returns:
            Tuple of (resistance, support)
        """
        resistance = high.rolling(window).max()
        support = low.rolling(window).min()
        
        return resistance, support
    
    @staticmethod
    def risk_reward_ratio(current_price: float, recent_high: float, recent_low: float) -> Dict[str, float]:
        """
        Calculate risk/reward ratio based on current price and recent high/low levels
        
        Args:
            current_price: Current stock price
            recent_high: Recent high price (resistance level)
            recent_low: Recent low price (support level)
            
        Returns:
            Dictionary containing risk, reward, ratio, and status information
        """
        # Calculate risk and reward
        risk = current_price - recent_low
        reward = recent_high - current_price
        ratio = reward / risk if risk > 0 else 0
        
        # Determine status based on ratio
        if ratio >= 2.0:
            status = "Favorable"
            score = 1.0
        elif ratio >= 1.0:
            status = "Balanced"
            score = 0.6
        else:
            status = "Unfavorable"
            score = 0.2
        
        return {
            'risk': risk,
            'reward': reward,
            'ratio': ratio,
            'status': status,
            'score': score
        }
    
    @staticmethod
    def support_resistance_levels(high: pd.Series, low: pd.Series, current_price: float, period: int = 20) -> Dict[str, any]:
        """
        Calculate support and resistance levels with distances
        
        Args:
            high: High price series
            low: Low price series
            current_price: Current stock price
            period: Period for calculating recent high/low (default 20)
            
        Returns:
            Dictionary containing support/resistance levels and distances
        """
        recent_high = high.tail(period).max()
        recent_low = low.tail(period).min()
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        
        # Calculate distances as percentages
        resistance_distance = ((recent_high - current_price) / current_price) * 100
        support_distance = ((current_price - recent_low) / current_price) * 100
        
        # Determine resistance status
        if resistance_distance > 10:
            resistance_status = "Distant"
        elif resistance_distance > 5:
            resistance_status = "Moderate"
        else:
            resistance_status = "Near"
        
        # Determine support status
        if support_distance > 5:
            support_status = "Strong"
        elif support_distance > 2:
            support_status = "Moderate"
        else:
            support_status = "Weak"
        
        return {
            'recent_high': recent_high,
            'recent_low': recent_low,
            'current_high': current_high,
            'current_low': current_low,
            'resistance_distance': resistance_distance,
            'support_distance': support_distance,
            'resistance_status': resistance_status,
            'support_status': support_status
        }
    
    @staticmethod
    def calc_slope(series: pd.Series, window: int = 10) -> float:
        """
        Calculate the slope of a series using linear regression
        
        Args:
            series: Data series
            window: Window for calculation
            
        Returns:
            Slope value
        """
        if len(series) < 2:
            return 0.0
        
        # Use last 'window' points or all available if less than window
        data = series.tail(window).dropna()
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        y = data.values
        
        # Calculate slope using least squares
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) * np.sum(x))
        
        return slope


def calculate_comprehensive_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive technical indicators for stock data.
    This function combines all technical indicators into one calculation.
    
    Args:
        df: DataFrame with columns: CLOSE, VOL, HIGH, LOW, OPEN (uppercase)
        
    Returns:
        DataFrame with all technical indicators added
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Ensure we have the required base columns
    required_cols = ['CLOSE', 'VOL', 'HIGH', 'LOW', 'OPEN']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Available columns: {list(df.columns)}")
    
    # =================
    # BASIC PRICE FEATURES
    # =================
    df['PRICE_CHANGE'] = df['CLOSE'].pct_change()
    df['PRICE_CHANGE_ABS'] = df['PRICE_CHANGE'].abs()
    df['HIGH_LOW_RATIO'] = df['HIGH'] / df['LOW']
    df['OPEN_CLOSE_RATIO'] = df['OPEN'] / df['CLOSE']
    df['PRICE_VOLATILITY'] = (df['HIGH'] - df['LOW']) / df['CLOSE']
    
    # =================
    # MOVING AVERAGES
    # =================
    # Simple Moving Averages
    for window in [5, 20, 50, 144]:
        col_name = f'SMA_{window}'
        df[col_name] = TechnicalIndicators.sma(df['CLOSE'], window)
        df[f'PRICE_TO_SMA_{window}'] = df['CLOSE'] / df[col_name]
        df[f'SMA_{window}_SLOPE'] = df[col_name].rolling(window=20).apply(
            lambda x: TechnicalIndicators.calc_slope(pd.Series(x)), raw=False
        )
    
    # SMA_144 specific features
    df['SMA_144_Dist'] = (df['CLOSE'] - df['SMA_144']) / df['SMA_144'] * 100
    
    # Exponential Moving Averages
    df['EMA_12'] = TechnicalIndicators.ema(df['CLOSE'], 12)
    df['EMA_26'] = TechnicalIndicators.ema(df['CLOSE'], 26)
    
    # =================
    # MACD FAMILY
    # =================
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = TechnicalIndicators.macd(df['CLOSE'])
    df['MACD_Up'] = df['MACD_Hist'].diff() > 0
    
    # Enhanced MACD features
    df['MACD_HISTOGRAM'] = df['MACD_Hist']  # Alias for consistency
    df['MACD_MOMENTUM'] = df['MACD'].pct_change(3)
    
    # =================
    # RSI
    # =================
    df['RSI_14'] = TechnicalIndicators.rsi(df['CLOSE'], 14)
    
    # =================
    # BOLLINGER BANDS
    # =================
    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = TechnicalIndicators.bollinger_bands(df['CLOSE'], 20, 2)
    df['BB_Std'] = df['CLOSE'].rolling(20).std(ddof=0)
    df['Touches_Lower_BB'] = df['CLOSE'] <= df['BB_Lower']
    df['Touches_Upper_BB'] = df['CLOSE'] >= df['BB_Upper']
    
    # Enhanced Bollinger Bands features
    df['BB_WIDTH'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_POSITION'] = (df['CLOSE'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # =================
    # STOCHASTIC OSCILLATOR
    # =================
    df['STOCH_K'], df['STOCH_D'] = TechnicalIndicators.stochastic(df['HIGH'], df['LOW'], df['CLOSE'])
    
    # =================
    # VOLUME INDICATORS
    # =================
    # Basic volume indicators
    df['Vol_SMA_10'] = TechnicalIndicators.sma(df['VOL'], 10)
    df['Volume_Spike'] = df['VOL'] > 1.5 * df['Vol_SMA_10']
    
    # Enhanced volume features
    df['VOLUME_SMA'] = TechnicalIndicators.sma(df['VOL'], 20)
    df['VOLUME_RATIO'] = df['VOL'] / df['VOLUME_SMA']
    df['VOLUME_MOMENTUM'] = df['VOL'].pct_change(3)
    df['PRICE_VOLUME'] = df['PRICE_CHANGE'] * df['VOLUME_RATIO']
    
    # On-Balance Volume (OBV)
    df['OBV'] = TechnicalIndicators.obv(df['CLOSE'], df['VOL'])
    df['OBV_MOMENTUM'] = df['OBV'].pct_change(5)
    
    # =================
    # VOLATILITY MEASURES
    # =================
    df['VOLATILITY_20'] = TechnicalIndicators.volatility(df['PRICE_CHANGE'], 20)
    df['VOLATILITY_MOMENTUM'] = df['VOLATILITY_20'].pct_change(5)
    
    # =================
    # SUPPORT/RESISTANCE LEVELS
    # =================
    df['RESISTANCE_20'], df['SUPPORT_20'] = TechnicalIndicators.support_resistance(df['HIGH'], df['LOW'], 20)
    df['RESISTANCE_DISTANCE'] = (df['RESISTANCE_20'] - df['CLOSE']) / df['CLOSE']
    df['SUPPORT_DISTANCE'] = (df['CLOSE'] - df['SUPPORT_20']) / df['CLOSE']
    
    # Calculate Risk/Reward ratios for each row (using rolling recent high/low)
    df['RECENT_HIGH_20'] = df['HIGH'].rolling(20).max()
    df['RECENT_LOW_20'] = df['LOW'].rolling(20).min()
    df['RISK_AMOUNT'] = df['CLOSE'] - df['RECENT_LOW_20']
    df['REWARD_AMOUNT'] = df['RECENT_HIGH_20'] - df['CLOSE']
    df['RISK_REWARD_RATIO'] = df['REWARD_AMOUNT'] / df['RISK_AMOUNT'].replace(0, np.nan)
    
    # =================
    # PRICE DISTANCE TO MA
    # =================
    df['PMA_FAST'], df['PMA_FAST_SIGNAL'], df['PMA_FAST_CYCLE'] = TechnicalIndicators.price_distance_to_ma(
        df['CLOSE'], ma_length=20, signal_length=9, exponential=False
    )
    
    # PMA Threshold Bands
    pma_bands = TechnicalIndicators.pma_threshold_bands(df['PMA_FAST'])
    df['PMA_UPPER_LOW'] = pma_bands['upper_low']
    df['PMA_LOWER_LOW'] = pma_bands['lower_low']
    df['PMA_UPPER_HIGH'] = pma_bands['upper_high']
    df['PMA_LOWER_HIGH'] = pma_bands['lower_high']
    
    return df


# Convenience functions for backwards compatibility
def sma(data: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average - convenience function"""
    return TechnicalIndicators.sma(data, window)


def ema(data: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average - convenience function"""
    return TechnicalIndicators.ema(data, window)


def rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """RSI - convenience function"""
    return TechnicalIndicators.rsi(data, period)


def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD - convenience function"""
    return TechnicalIndicators.macd(data, fast, slow, signal)


def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands - convenience function"""
    return TechnicalIndicators.bollinger_bands(data, window, std_dev)


def risk_reward_ratio(current_price: float, recent_high: float, recent_low: float) -> Dict[str, float]:
    """Risk/Reward Ratio - convenience function"""
    return TechnicalIndicators.risk_reward_ratio(current_price, recent_high, recent_low)


def support_resistance_levels(high: pd.Series, low: pd.Series, current_price: float, period: int = 20) -> Dict[str, any]:
    """Support/Resistance Levels - convenience function"""
    return TechnicalIndicators.support_resistance_levels(high, low, current_price, period)