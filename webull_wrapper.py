from webull import webull, paper_webull
import time
from webull_config import EMAIL, PHONE, PASSWORD, DEVICE_NAME, USE_PAPER, LOGIN_METHOD, TRADE_TOKEN, get_2fa_code

class WebullAPI:
    def __init__(self):
        self.wb = paper_webull() if USE_PAPER else webull()
        self.account_id = None

    def login(self):
        print("[+] Logging in...")
        try:
            # Try refreshing existing session first (avoids 2FA if session still valid)
            print("[+] Checking for saved session...")
            try:
                did_refresh = self.wb.refresh_login()
            except Exception as refresh_error:
                print(f"[!] Session refresh failed: {refresh_error}")
                did_refresh = False

            if did_refresh:
                print("[+] ✅ Used saved session (no 2FA needed)")
                self.account_id = self.wb.get_account()['secAccountId']
                print("[+] Login success. Account:", self.account_id)
                return
            else:
                print("[+] No saved session. Requesting 2FA...")

            # Full login with 2FA
            print("[+] Requesting 2FA code from Webull...")

            # Determine login identifier based on method
            if LOGIN_METHOD == 'phone' and PHONE:
                login_id = PHONE
                print(f"[+] Using phone number: ***-***-{PHONE[-4:]}")
            else:
                login_id = EMAIL
                print(f"[+] Using email: {EMAIL[:3]}***@{EMAIL.split('@')[1]}")

            try:
                mfa_response = self.wb.get_mfa(login_id)
                print(f"[+] MFA request sent. Response: {mfa_response}")
                print(f"[+] Check your phone {PHONE[-4:] if LOGIN_METHOD == 'phone' else 'for SMS'}")
            except Exception as mfa_error:
                print(f"[!] Failed to request MFA: {mfa_error}")
                raise

            code = get_2fa_code()

            if not code or not code.strip():
                raise ValueError("No 2FA code provided")

            print(f"[+] Device name: {DEVICE_NAME}")

            login_response = self.wb.login(
                login_id,
                PASSWORD,
                device_name=DEVICE_NAME,
                mfa=code
            )

            print(f"[+] Login response: {login_response}")

            self.account_id = self.wb.get_account()['secAccountId']
            print("[+] Login success. Account:", self.account_id)
            print("[+] Session saved for future logins")

            # Get trade token if not already set
            self._ensure_trade_token()

        except ValueError as ve:
            print(f"[!] Input error: {ve}")
            exit(1)
        except KeyError as ke:
            print(f"[!] API response missing expected field: {ke}")
            print("[!] This may indicate:")
            print("    - Invalid credentials (wrong email/password)")
            print("    - Incorrect or expired 2FA code")
            print("    - Account locked or requires verification")
            exit(1)
        except Exception as e:
            import traceback
            error_type = type(e).__name__
            print(f"[!] Login failed ({error_type}): {e}")

            # Check for specific error types
            if "Expecting value" in str(e):
                print("\n[!] JSON Parse Error - Webull returned invalid response")
                print("    Common causes:")
                print("    1. Wrong email or password")
                print("    2. 2FA code incorrect or expired")
                print("    3. Account needs verification on Webull website")
                print("    4. Too many failed login attempts (wait 15 min)")
                print("\n    Try:")
                print("    - Login to Webull website manually first")
                print("    - Verify credentials in .env are correct")
                print("    - Use a fresh 2FA code (request new one)")

            print("\nFull error details:")
            traceback.print_exc()
            print("\nTroubleshooting tips:")
            print("1. Check SMS on your phone for 2FA code")
            print("2. Open Webull mobile app for the code")
            print("3. Verify credentials in .env are correct")
            print("4. Try logging into https://www.webull.com manually")
            print("5. See WEBULL_2FA_GUIDE.md for detailed help")
            exit(1)

    def _ensure_trade_token(self):
        """Ensure we have a valid trade token for placing orders"""
        try:
            # Check if we already have a trade token from env
            if TRADE_TOKEN and TRADE_TOKEN.strip():
                print(f"[+] Using trade token from .env: {TRADE_TOKEN[:3]}***")
                return

            # Get trade token via 2FA
            print("\n[!] Trade token required for placing orders")
            print("[+] Requesting trade token...")

            # Determine login identifier
            if LOGIN_METHOD == 'phone' and PHONE:
                login_id = PHONE
            else:
                login_id = EMAIL

            # Request trade token
            trade_token_response = self.wb.get_trade_token(login_id)
            print(f"[+] Trade token request sent: {trade_token_response}")

            # Get code from user
            print("\n📱 Check your phone for the trade PIN (6 digits)")
            trade_pin = input("Enter trade PIN: ").strip()

            if not trade_pin:
                print("[!] No trade PIN entered - orders may fail")
                return

            # Set the trade token
            self.wb.get_trade_token(login_id, trade_pin)
            print("[+] ✅ Trade token set successfully")
            print(f"[!] Save this token to .env as WEBULL_TRADE_TOKEN={trade_pin}")

        except Exception as e:
            print(f"[!] Failed to get trade token: {e}")
            print("[!] You may not be able to place orders without it")

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

    def get_current_orders(self):
        """Get current open/pending orders"""
        try:
            return self.wb.get_current_orders()
        except Exception as e:
            print(f"[!] Failed to get current orders: {e}")
            return []

