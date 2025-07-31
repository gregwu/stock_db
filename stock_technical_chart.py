#!/usr/bin/env python3
"""
Stock Chart with Technical Indicators
Includes: Bollinger Bands, Volume, SMA 144, and Price Distance to MA (converted from Pine Script)

Features:
- Candlestick chart with volume
- Bollinger Bands (20-period)
- SMA 144 trend line
- Price Distance to MA indicator (converted from PriceDistMA.pine)
- Interactive charts using plotly
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import argparse

class TechnicalIndicators:
    """Calculate various technical indicators"""
    
    @staticmethod
    def bollinger_bands(data, window=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return sma, upper_band, lower_band
    
    @staticmethod
    def sma(data, window):
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def price_distance_to_ma(price, ma_length=20, signal_length=9, exponential=False):
        """
        Price Distance to Moving Average indicator (converted from Pine Script)
        
        Args:
            price: Price series (typically close price)
            ma_length: Moving average length (default 20)
            signal_length: Signal line length (default 9)
            exponential: Use EMA instead of SMA (default False)
        
        Returns:
            pma: Price/MA ratio as percentage
            signal: Signal line (MA of PMA)
            cycle: Cycle histogram (PMA - Signal)
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
    def pma_threshold_bands(pma, bb_length=200, std_dev_low=1.5, std_dev_high=2.25):
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

class StockChartPlotter:
    """Create interactive stock charts with technical indicators"""
    
    def __init__(self, symbol, period='1y', plot_period='6mo'):
        """
        Initialize with stock symbol and data period
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            period: Data fetch period - always use at least 1y for SMA 144 calculation
            plot_period: Period to display in chart ('3mo', '6mo', '1y', etc.)
        """
        self.symbol = symbol.upper()
        # Ensure we have enough data for SMA 144 (minimum 1 year)
        period_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, 
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650
        }
        
        # If requested period is less than 1y, fetch 1y but keep original for plotting
        if period in period_days and period_days[period] < 365:
            self.period = '1y'  # Fetch 1 year for SMA 144 calculation
            self.plot_period = period  # But only plot the requested period
        else:
            self.period = period
            self.plot_period = plot_period
            
        self.data = None
        self.plot_data = None
        self.load_data()
    
    def load_data(self):
        """Load stock data from Yahoo Finance"""
        try:
            print(f"📊 Loading data for {self.symbol} (fetch period: {self.period}, plot period: {self.plot_period})...")
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            # Create plot data subset based on plot_period
            plot_days = {
                '1d': 1, '5d': 5, '1mo': 22, '3mo': 66, '6mo': 132, 
                '1y': 252, '2y': 504, '5y': 1260, '10y': 2520
            }
            
            if self.plot_period in plot_days:
                days_to_plot = plot_days[self.plot_period]
                self.plot_data = self.data.tail(days_to_plot).copy()
            else:
                self.plot_data = self.data.copy()
            
            print(f"✅ Loaded {len(self.data)} trading days (plotting last {len(self.plot_data)} days)")
            print(f"📅 Full data range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            print(f"📊 Plot data range: {self.plot_data.index[0].date()} to {self.plot_data.index[-1].date()}")
            
        except Exception as e:
            print(f"❌ Error loading data for {self.symbol}: {e}")
            raise
    
    def calculate_indicators(self):
        """Calculate all technical indicators using full data, return subset for plotting"""
        # Use full data for calculations to ensure SMA 144 has enough data points
        close = self.data['Close']
        
        # Bollinger Bands (20-period, 2 std dev)
        bb_sma, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(close, 20, 2)
        
        # SMA 144 - calculated on full dataset
        sma_144 = TechnicalIndicators.sma(close, 144)
        
        # Price Distance to MA (Fast: 20-period)
        pma_fast, pma_fast_signal, pma_fast_cycle = TechnicalIndicators.price_distance_to_ma(
            close, ma_length=20, signal_length=9, exponential=False
        )
        
        # PMA Threshold Bands
        pma_bands = TechnicalIndicators.pma_threshold_bands(pma_fast)
        
        # Create indicators dataframe for full data
        full_indicators = pd.DataFrame({
            'bb_sma': bb_sma,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'sma_144': sma_144,
            'pma_fast': pma_fast,
            'pma_fast_signal': pma_fast_signal,
            'pma_fast_cycle': pma_fast_cycle,
            'pma_upper_low': pma_bands['upper_low'],
            'pma_lower_low': pma_bands['lower_low'],
            'pma_upper_high': pma_bands['upper_high'],
            'pma_lower_high': pma_bands['lower_high']
        }, index=self.data.index)
        
        # Return subset that matches plot_data timeframe
        plot_indicators = full_indicators.loc[self.plot_data.index]
        
        return plot_indicators
    
    def create_chart(self):
        """Create comprehensive stock chart with all indicators merged into one chart"""
        indicators = self.calculate_indicators()
        
        # Create a single chart with secondary y-axes
        fig = go.Figure()
        
        # 1. Main price chart with candlesticks (using plot_data)
        # Calculate daily changes for hover information
        daily_changes_pct = self.plot_data['Close'].pct_change() * 100
        daily_changes_dollar = self.plot_data['Close'].diff()
        
        # Create custom hover text with price change information
        hover_text = []
        for i, (date, row) in enumerate(self.plot_data.iterrows()):
            if i == 0:  # First day has no previous day for comparison
                change_text = "N/A"
            else:
                pct_change = daily_changes_pct.iloc[i]
                dollar_change = daily_changes_dollar.iloc[i]
                change_text = f"{pct_change:+.2f}% (${dollar_change:+.2f})"
            
            hover_text.append(
                f"Date: {date.strftime('%Y-%m-%d')}<br>"
                f"Open: ${row['Open']:.2f}<br>"
                f"High: ${row['High']:.2f}<br>"
                f"Low: ${row['Low']:.2f}<br>"
                f"Close: ${row['Close']:.2f}<br>"
                f"Volume: {row['Volume']:,.0f}<br>"
                f"Daily Change: {change_text}"
            )
        
        fig.add_trace(
            go.Candlestick(
                x=self.plot_data.index,
                open=self.plot_data['Open'],
                high=self.plot_data['High'],
                low=self.plot_data['Low'],
                close=self.plot_data['Close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                hovertext=hover_text,
                hoverinfo='text'
            )
        )
        
        # 2. Bollinger Bands (using plot_data index)
        fig.add_trace(
            go.Scatter(
                x=self.plot_data.index,
                y=indicators['bb_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(173, 216, 230, 0.8)', width=1),
                hovertemplate='BB Upper: $%{y:.2f}<extra></extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.plot_data.index,
                y=indicators['bb_sma'],
                mode='lines',
                name='BB Middle (SMA 20)',
                line=dict(color='orange', width=2),
                hovertemplate='BB Middle: $%{y:.2f}<extra></extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.plot_data.index,
                y=indicators['bb_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(173, 216, 230, 0.8)', width=1),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.1)',
                hovertemplate='BB Lower: $%{y:.2f}<extra></extra>'
            )
        )
        
        # 3. SMA 144 trend line
        fig.add_trace(
            go.Scatter(
                x=self.plot_data.index,
                y=indicators['sma_144'],
                mode='lines',
                name='SMA 144 (Trend)',
                line=dict(color='purple', width=3),
                hovertemplate='SMA 144: $%{y:.2f}<extra></extra>'
            )
        )
        
        # 4. Volume chart (using secondary y-axis)
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(self.plot_data['Close'], self.plot_data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=self.plot_data.index,
                y=self.plot_data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.3,
                yaxis='y2',
                hovertemplate='Volume: %{y:,.0f}<extra></extra>'
            )
        )
        
        # 5. Price Distance to MA indicator (converted from Pine Script)
        
        # PMA threshold bands (background fills) - using plot_data index
        fig.add_trace(
            go.Scatter(
                x=self.plot_data.index,
                y=indicators['pma_upper_high'],
                mode='lines',
                name='Overbought High',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                showlegend=False,
                yaxis='y3'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.plot_data.index,
                y=indicators['pma_upper_low'],
                mode='lines',
                name='Overbought Low',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.15)',
                showlegend=False,
                yaxis='y3'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.plot_data.index,
                y=indicators['pma_lower_low'],
                mode='lines',
                name='Oversold High',
                line=dict(color='rgba(0, 255, 0, 0.3)', width=1),
                showlegend=False,
                yaxis='y3'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.plot_data.index,
                y=indicators['pma_lower_high'],
                mode='lines',
                name='Oversold Low',
                line=dict(color='rgba(0, 255, 0, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(0, 255, 0, 0.15)',
                showlegend=False,
                yaxis='y3'
            )
        )
        
        # PMA Fast line
        fig.add_trace(
            go.Scatter(
                x=self.plot_data.index,
                y=indicators['pma_fast'],
                mode='lines',
                name='Price/MA %',
                line=dict(color='red', width=2),
                hovertemplate='Price/MA: %{y:.2f}%<extra></extra>',
                yaxis='y3'
            )
        )
        
        # PMA Signal line
        fig.add_trace(
            go.Scatter(
                x=self.plot_data.index,
                y=indicators['pma_fast_signal'],
                mode='lines',
                name='Signal Line',
                line=dict(color='blue', width=2),
                hovertemplate='Signal: %{y:.2f}%<extra></extra>',
                yaxis='y3'
            )
        )
        
        # Cycle histogram (as bar chart)
        cycle_colors = ['green' if x > 0 else 'red' for x in indicators['pma_fast_cycle']]
        fig.add_trace(
            go.Bar(
                x=self.plot_data.index,
                y=indicators['pma_fast_cycle'],
                name='Cycle Histogram',
                marker_color=cycle_colors,
                opacity=0.6,
                hovertemplate='Cycle: %{y:.2f}%<extra></extra>',
                yaxis='y3'
            )
        )
        
        # Update layout with enhanced crosshair configuration
        fig.update_layout(
            title=f'{self.symbol} - Comprehensive Technical Analysis',
           # xaxis_title='Date',
            template='plotly_dark',
            height=1000,
            showlegend=True,
            hovermode='x unified',
            # Enhanced crosshair configuration
            hoverdistance=100,
            spikedistance=1000,
            # Main x-axis (shared across all chart sections)
            xaxis=dict(
                showspikes=True,
                spikecolor="rgba(255,255,255,0.8)",
                spikesnap="cursor",
                spikemode="across",
                spikethickness=2,
                spikedash="solid",
                domain=[0.0, 1.0],
                anchor='y'
            ),
            # Primary y-axis for price data
            yaxis=dict(
                title='Price ($)',
                side='right',
                domain=[0.4, 1.0]  # Price chart takes top 60% of space
            ),
            # Secondary y-axis for volume (middle section)
            yaxis2=dict(
                title='Volume',
                side='left',
                showgrid=False,
                domain=[0.28, 0.38]  # Volume chart takes middle 10% of space
            ),
            # Third y-axis for PMA indicator (bottom section)
            yaxis3=dict(
                title='Price/MA (%)',
                side='right',
                showgrid=True,
                domain=[0.0, 0.25],  # PMA indicator takes bottom 25% of space
                zeroline=True,
                zerolinecolor='gray',
                zerolinewidth=1
            ),
            # Add shapes for crosshair spanning both sections
            shapes=[
                # Vertical crosshair line spanning entire chart height
                dict(
                    type="line",
                    x0=0, x1=0,
                    y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color="rgba(255,255,255,0.8)", width=2, dash="solid"),
                    visible=False
                )
            ]
        )
        
        # Remove rangeslider and hide weekends/non-trading days
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        # Apply enhanced crosshair settings and remove weekends/non-trading days
        fig.update_xaxes(
            showspikes=True,
            spikecolor="rgba(255,255,255,0.8)",
            spikesnap="cursor",
            spikemode="across",
            spikethickness=2,
            spikedash="solid",
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends (Saturday and Sunday)
                dict(values=["2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04", "2023-10-09", "2023-11-23", "2023-12-25"]),  # US holidays 2023
                dict(values=["2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-28", "2024-12-25"]),  # US holidays 2024
                dict(values=["2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-27", "2025-12-25"]),  # US holidays 2025
            ]
        )
        
        return fig
    
    def save_chart(self, filename=None):
        """Save chart as HTML file with enhanced crosshair"""
        if filename is None:
            filename = f'{self.symbol}_technical_analysis_{datetime.now().strftime("%Y%m%d")}.html'
        
        fig = self.create_chart()
        
        # Custom JavaScript for enhanced crosshair spanning both chart sections
        custom_js = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            
            if (plotDiv && plotDiv.on) {
                plotDiv.on('plotly_hover', function(data) {
                    var xval = data.points[0].x;
                    var update = {
                        'shapes[0].x0': xval,
                        'shapes[0].x1': xval,
                        'shapes[0].visible': true
                    };
                    Plotly.relayout(plotDiv, update);
                });
                
                plotDiv.on('plotly_unhover', function() {
                    var update = {
                        'shapes[0].visible': false
                    };
                    Plotly.relayout(plotDiv, update);
                });
            }
        });
        </script>
        """
        
        # Save the chart with custom JavaScript
        html_string = fig.to_html(
            include_plotlyjs=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{self.symbol}_technical_chart',
                    'height': 1000,
                    'width': 1400,
                    'scale': 2
                }
            }
        )
        
        # Inject custom JavaScript into the HTML
        html_string = html_string.replace('</body>', custom_js + '</body>')
        
        # Write the modified HTML to file
        with open(filename, 'w') as f:
            f.write(html_string)
        
        print(f"💾 Chart saved as: {filename}")
        return filename
    
    def show_chart(self):
        """Display interactive chart with enhanced crosshair"""
        fig = self.create_chart()
        
        # Configure for better crosshair visibility in live mode
        fig.update_layout(
            # Ensure spikes are visible and properly configured
            hoverdistance=50,
            spikedistance=1000
        )
        
        # Show with enhanced configuration
        fig.show(config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{self.symbol}_technical_chart',
                'height': 1000,
                'width': 1400,
                'scale': 2
            }
        })
    
    def get_latest_signals(self):
        """Get latest trading signals from indicators"""
        # Calculate indicators using full data
        close = self.data['Close']
        bb_sma, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(close, 20, 2)
        sma_144 = TechnicalIndicators.sma(close, 144)
        pma_fast, pma_fast_signal, pma_fast_cycle = TechnicalIndicators.price_distance_to_ma(
            close, ma_length=20, signal_length=9, exponential=False
        )
        
        # Get latest values
        latest_close = self.data['Close'].iloc[-1]
        latest_bb_upper = bb_upper.iloc[-1] if not bb_upper.isna().iloc[-1] else None
        latest_bb_lower = bb_lower.iloc[-1] if not bb_lower.isna().iloc[-1] else None
        latest_sma_144 = sma_144.iloc[-1] if not sma_144.isna().iloc[-1] else None
        latest_pma_fast = pma_fast.iloc[-1] if not pma_fast.isna().iloc[-1] else None
        latest_pma_signal = pma_fast_signal.iloc[-1] if not pma_fast_signal.isna().iloc[-1] else None
        latest_pma_cycle = pma_fast_cycle.iloc[-1] if not pma_fast_cycle.isna().iloc[-1] else None
        
        # Calculate daily change percentage
        if len(self.data) >= 2:
            previous_close = self.data['Close'].iloc[-2]
            daily_change_pct = ((latest_close - previous_close) / previous_close) * 100
            daily_change_dollars = latest_close - previous_close
            formatted_daily_change = f"{daily_change_pct:+.2f}% (${daily_change_dollars:+.2f})"
        else:
            daily_change_pct = None
            daily_change_dollars = None
            formatted_daily_change = "N/A"
        
        signals = {
            'symbol': self.symbol,
            'date': self.data.index[-1].strftime('%Y-%m-%d'),
            'close_price': latest_close,
            'daily_change_pct': daily_change_pct,
            'daily_change_dollars': daily_change_dollars,
            'formatted_daily_change': formatted_daily_change,
            'bb_position': 'Above Upper' if latest_bb_upper and latest_close > latest_bb_upper else 
                          'Below Lower' if latest_bb_lower and latest_close < latest_bb_lower else 'Within Bands',
            'sma_144_trend': 'Above' if latest_sma_144 and latest_close > latest_sma_144 else 'Below' if latest_sma_144 else 'N/A',
            'pma_fast': latest_pma_fast,
            'pma_signal': latest_pma_signal,
            'pma_cycle': latest_pma_cycle,
            'pma_trend': 'Bullish' if latest_pma_fast and latest_pma_signal and latest_pma_fast > latest_pma_signal else 'Bearish'
        }
        
        return signals

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Stock Technical Analysis Chart')
    parser.add_argument('symbol', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--period', default='1y', 
                       choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
                       help='Data fetch period - minimum 1y for SMA 144 (default: 1y)')
    parser.add_argument('--plot-period', default='6mo',
                       choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'],
                       help='Period to display in chart (default: 6mo)')
    parser.add_argument('--save', action='store_true', help='Save chart as HTML file')
    parser.add_argument('--filename', help='Custom filename for saved chart')
    parser.add_argument('--signals', action='store_true', help='Show latest trading signals')
    
    args = parser.parse_args()
    
    try:
        # Create chart plotter
        plotter = StockChartPlotter(args.symbol, args.period, args.plot_period)
        
        # Show signals if requested
        if args.signals:
            signals = plotter.get_latest_signals()
            print(f"\n📊 LATEST SIGNALS for {signals['symbol']} ({signals['date']}):")
            print(f"   💰 Price: ${signals['close_price']:.2f}")
            print(f"   📈 Daily Change: {signals['formatted_daily_change']}")
            print(f"   📊 BB Position: {signals['bb_position']}")
            print(f"   � SMA 144 Trend: {signals['sma_144_trend']}")
            print(f"   🎯 Price/MA: {signals['pma_fast']:.2f}%" if signals['pma_fast'] else "   🎯 Price/MA: N/A")
            print(f"   📡 Signal Line: {signals['pma_signal']:.2f}%" if signals['pma_signal'] else "   📡 Signal Line: N/A")
            print(f"   🔄 Cycle: {signals['pma_cycle']:.2f}%" if signals['pma_cycle'] else "   🔄 Cycle: N/A")
            print(f"   📈 PMA Trend: {signals['pma_trend']}")
        
        # Save or show chart
        if args.save:
            plotter.save_chart(args.filename)
        else:
            print("🚀 Opening interactive chart...")
            plotter.show_chart()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
