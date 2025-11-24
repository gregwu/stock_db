from webull import webull, paper_webull
import time
from webull_config import EMAIL, PASSWORD, DEVICE_NAME, USE_PAPER, get_2fa_code

class WebullAPI:
    def __init__(self):
        self.wb = paper_webull() if USE_PAPER else webull()
        self.account_id = None

    def login(self):
        print("[+] Logging in...")
        try:
            # request 2FA challenge
            self.wb.get_mfa(EMAIL)
            code = get_2fa_code()

            self.wb.login(
                EMAIL,
                PASSWORD,
                device_name=DEVICE_NAME,
                mfa=code
            )

            self.account_id = self.wb.get_account()['secAccountId']
            print("[+] Login success. Account:", self.account_id)

        except Exception as e:
            print("[!] Login failed:", e)
            exit(1)

    def quote(self, ticker):
        return self.wb.get_quote(ticker)

    def place_order(self, ticker, qty, action, order_type="MKT", price=0):
        print(f"[+] Placing {action} order for {qty} {ticker}")

        order = self.wb.place_order(
            stock=self.wb.get_stock(ticker),
            price=price,
            quant=qty,
            action=action,
            orderType=order_type,
            enforce="DAY",
            outsideRegularTradingHour=True
        )
        print("[+] Order result:", order)
        return order

    def cancel(self, order_id):
        try:
            return self.wb.cancel_order(order_id)
        except:
            return None

    def get_positions(self):
        """Get current portfolio positions"""
        try:
            return self.wb.get_positions()
        except Exception as e:
            print(f"[!] Failed to get positions: {e}")
            return []

    def get_portfolio(self):
        """Get full portfolio summary"""
        try:
            return self.wb.get_portfolio()
        except Exception as e:
            print(f"[!] Failed to get portfolio: {e}")
            return {}

