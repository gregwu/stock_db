"""
Alpaca API Wrapper for Trading
Provides a simple interface to Alpaca's trading API
"""
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class AlpacaAPI:
    def __init__(self, paper=True):
        """
        Initialize Alpaca API client

        Args:
            paper: If True, use paper trading account. If False, use live account.
        """
        # Use different environment variables for paper vs live accounts
        if paper:
            self.api_key = os.getenv('ALPACA_API_KEY_PAPER')
            self.secret_key = os.getenv('ALPACA_SECRET_KEY_PAPER')
            key_type = "ALPACA_API_KEY_PAPER and ALPACA_SECRET_KEY_PAPER"
        else:
            self.api_key = os.getenv('ALPACA_API_KEY')
            self.secret_key = os.getenv('ALPACA_SECRET_KEY')
            key_type = "ALPACA_API_KEY and ALPACA_SECRET_KEY"

        if not self.api_key or not self.secret_key:
            raise ValueError(f"{key_type} must be set in .env file")

        # Initialize trading client
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=paper
        )

        # Initialize data client (for quotes)
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

        self.paper = paper
        self.account_id = None

        print(f"[+] Alpaca API initialized ({'PAPER' if paper else 'LIVE'} account)")

    def login(self):
        """
        Verify connection and get account info
        No login required for Alpaca - API keys are used directly
        """
        try:
            account = self.trading_client.get_account()
            self.account_id = account.id

            print(f"[+] Connected to Alpaca")
            print(f"[+] Account ID: {self.account_id}")
            print(f"[+] Buying Power: ${float(account.buying_power):,.2f}")
            print(f"[+] Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"[+] Cash: ${float(account.cash):,.2f}")

            return True

        except Exception as e:
            print(f"[!] Failed to connect to Alpaca: {e}")
            return False

    def get_account(self):
        """Get account information"""
        try:
            return self.trading_client.get_account()
        except Exception as e:
            print(f"[!] Failed to get account: {e}")
            return None

    def quote(self, ticker):
        """Get current quote for a ticker"""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quotes = self.data_client.get_stock_latest_quote(request)
            quote = quotes[ticker]

            return {
                'symbol': ticker,
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'last': (float(quote.bid_price) + float(quote.ask_price)) / 2,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size
            }
        except Exception as e:
            print(f"[!] Failed to get quote for {ticker}: {e}")
            return None

    def get_bars(self, ticker, interval='5m', period='1d', feed='iex'):
        """
        Get historical bars from Alpaca

        Args:
            ticker: Stock symbol
            interval: Bar interval - '1m', '5m', '15m', '30m', '1h', '1d'
            period: How far back to fetch - '1d', '5d', '1mo', etc.
            feed: Data feed - 'iex' (free, 15-min delayed) or 'sip' (paid, real-time)

        Returns:
            pandas DataFrame with OHLCV data, index as timestamp in Eastern Time
        """
        try:
            # Map interval string to Alpaca TimeFrame
            timeframe_map = {
                '1m': TimeFrame.Minute,
                '5m': TimeFrame(5, TimeFrameUnit.Minute),
                '15m': TimeFrame(15, TimeFrameUnit.Minute),
                '30m': TimeFrame(30, TimeFrameUnit.Minute),
                '1h': TimeFrame.Hour,
                '1d': TimeFrame.Day
            }

            if interval not in timeframe_map:
                raise ValueError(f"Unsupported interval: {interval}. Use: {list(timeframe_map.keys())}")

            timeframe = timeframe_map[interval]

            # Convert period to start/end dates
            end = datetime.now()
            if period == '1d':
                start = end - timedelta(days=1)
            elif period == '5d':
                start = end - timedelta(days=5)
            elif period == '1wk':
                start = end - timedelta(weeks=1)
            elif period == '1mo':
                start = end - timedelta(days=30)
            elif period == '3mo':
                start = end - timedelta(days=90)
            else:
                # Default to 1 day
                start = end - timedelta(days=1)

            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=timeframe,
                start=start,
                end=end,
                feed=feed  # 'iex' for free (15-min delayed), 'sip' for paid (real-time)
            )

            # Fetch bars
            bars = self.data_client.get_stock_bars(request)

            if ticker not in bars or not bars[ticker]:
                print(f"[!] No bar data returned for {ticker}")
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for bar in bars[ticker]:
                data.append({
                    'Open': float(bar.open),
                    'High': float(bar.high),
                    'Low': float(bar.low),
                    'Close': float(bar.close),
                    'Volume': int(bar.volume),
                    'timestamp': bar.timestamp
                })

            df = pd.DataFrame(data)
            if df.empty:
                return df

            # Set timestamp as index and convert to Eastern Time
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Convert from UTC to Eastern Time
            if df.index.tz is not None:
                df.index = df.index.tz_convert('America/New_York')
            else:
                df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

            return df

        except Exception as e:
            print(f"[!] Failed to get bars for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def place_order(self, ticker, qty, action, order_type="LMT", price=None, extended_hours=True):
        """
        Place an order

        Args:
            ticker: Stock symbol
            qty: Number of shares
            action: "BUY" or "SELL"
            order_type: "LMT" for limit order (default), "MKT" for market order
            price: Limit price (required for limit orders)
            extended_hours: Enable extended hours trading (default: True)
        """
        import logging
        try:
            print(f"[+] Placing {action} order for {qty} {ticker}")
            logging.info(f"[AlpacaWrapper] Placing {action} order for {qty} {ticker}, order_type={order_type}, extended_hours={extended_hours}")

            # Determine order side
            side = OrderSide.BUY if action.upper() == "BUY" else OrderSide.SELL

            # Create order request
            if order_type == "MKT":
                order_data = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    extended_hours=extended_hours
                )
                logging.info(f"[AlpacaWrapper] Created MarketOrderRequest: symbol={ticker}, qty={qty}, side={side}")
            else:  # Limit order
                if price is None:
                    raise ValueError("Price must be specified for limit orders")
                order_data = LimitOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=price,
                    extended_hours=extended_hours
                )
                logging.info(f"[AlpacaWrapper] Created LimitOrderRequest: symbol={ticker}, qty={qty}, side={side}, price={price}")

            # Submit order
            logging.info(f"[AlpacaWrapper] Submitting order to Alpaca API...")
            order = self.trading_client.submit_order(order_data)
            logging.info(f"[AlpacaWrapper] Order submitted successfully")

            print(f"[+] Order placed successfully")
            print(f"[+] Order ID: {order.id}")
            print(f"[+] Status: {order.status}")

            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'filled_qty': order.filled_qty,
                'filled_avg_price': order.filled_avg_price,
                'limit_price': order.limit_price if hasattr(order, 'limit_price') else None
            }

        except Exception as e:
            import logging
            import traceback
            error_msg = f"Failed to place order for {ticker}: {type(e).__name__}: {e}"
            error_details = traceback.format_exc()
            print(f"[!] {error_msg}")
            print(f"[!] Full error details:\n{error_details}")
            logging.error(error_msg)
            logging.error(f"[AlpacaWrapper] Full traceback:\n{error_details}")
            # Return error info instead of None so it can be displayed in UI
            return {'error': str(e), 'error_type': type(e).__name__}

    def cancel(self, order_id):
        """Cancel an order"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            print(f"[+] Order {order_id} cancelled")
            return True
        except Exception as e:
            print(f"[!] Failed to cancel order {order_id}: {e}")
            return False

    def get_positions(self):
        """Get all open positions"""
        try:
            positions = self.trading_client.get_all_positions()

            result = []
            for pos in positions:
                result.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'side': pos.side,
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'avg_entry_price': float(pos.avg_entry_price)
                })

            return result

        except Exception as e:
            print(f"[!] Failed to get positions: {e}")
            return []

    def get_portfolio(self):
        """Get portfolio summary"""
        try:
            account = self.trading_client.get_account()

            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value)
            }

        except Exception as e:
            print(f"[!] Failed to get portfolio: {e}")
            return {}

    def get_current_orders(self):
        """Get all open/pending orders"""
        try:
            orders = self.trading_client.get_orders()

            result = []
            for order in orders:
                result.append({
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                    'side': order.side,
                    'type': order.type,
                    'status': order.status,
                    'created_at': order.created_at,
                    'limit_price': float(order.limit_price) if order.limit_price else None
                })

            return result

        except Exception as e:
            print(f"[!] Failed to get orders: {e}")
            return []

    def get_order(self, order_id):
        """Get a specific order by ID"""
        try:
            from alpaca.trading.requests import GetOrderByIdRequest

            order = self.trading_client.get_order_by_id(order_id)

            if order:
                return {
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                    'side': str(order.side),
                    'type': str(order.type),
                    'status': str(order.status),
                    'created_at': order.created_at,
                    'updated_at': order.updated_at,
                    'filled_at': order.filled_at,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None
                }
            return None

        except Exception as e:
            print(f"[!] Failed to get order {order_id}: {e}")
            return None

    def get_recent_filled_orders(self, limit=100):
        """Get recent filled orders"""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=limit
            )
            orders = self.trading_client.get_orders(filter=request)

            result = []
            for order in orders:
                if str(order.status) == 'OrderStatus.FILLED':
                    result.append({
                        'order_id': order.id,
                        'symbol': order.symbol,
                        'qty': float(order.qty),
                        'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                        'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                        'side': str(order.side).replace('OrderSide.', ''),
                        'type': str(order.type),
                        'status': str(order.status),
                        'created_at': order.created_at,
                        'updated_at': order.updated_at,
                        'filled_at': order.filled_at,
                        'limit_price': float(order.limit_price) if order.limit_price else None
                    })

            return result

        except Exception as e:
            print(f"[!] Failed to get recent filled orders: {e}")
            return []


if __name__ == "__main__":
    # Test the API
    print("Testing Alpaca API...")

    # Use paper trading for testing
    api = AlpacaAPI(paper=True)

    if api.login():
        print("\n[+] Getting account info...")
        account = api.get_account()
        if account:
            print(f"  Equity: ${float(account.equity):,.2f}")
            print(f"  Cash: ${float(account.cash):,.2f}")

        print("\n[+] Getting quote for TQQQ...")
        quote = api.quote('TQQQ')
        if quote:
            print(f"  Last: ${quote['last']:.2f}")
            print(f"  Bid: ${quote['bid']:.2f}")
            print(f"  Ask: ${quote['ask']:.2f}")

        print("\n[+] Getting positions...")
        positions = api.get_positions()
        if positions:
            for pos in positions:
                print(f"  {pos['symbol']}: {pos['qty']} shares @ ${pos['current_price']:.2f}")
                print(f"    P&L: ${pos['unrealized_pl']:.2f} ({pos['unrealized_plpc']*100:.2f}%)")
        else:
            print("  No open positions")

        print("\n[+] Getting open orders...")
        orders = api.get_current_orders()
        if orders:
            for order in orders:
                print(f"  {order['symbol']}: {order['side']} {order['qty']} shares - {order['status']}")
        else:
            print("  No open orders")

        print("\n✅ Alpaca API test completed successfully!")
    else:
        print("\n❌ Failed to connect to Alpaca")
