import sys
import logging
import os
import time
import math
import asyncio
import websockets
import json
import re
import pandas as pd
from threading import Thread
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from logging.handlers import RotatingFileHandler
import matplotlib

from apiBybit_past import (
    fetch_data_with_metrics,
    close_positions,
    get_account_balance,
    set_leverage,
    get_min_order_size,
    get_max_qty,
    get_open_position,
    switch_to_cross_margin,
    open_position_api,
    set_trading_stop,
    timeframe_to_interval,
    calculate_atr
)

from prediction_manager import PredictionManager

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

TIMEFRAME_CONFIG = CONFIG.get("timeframe_config", {
    "LONG_TF": {
        "use": True,
        "interval": "1h",
        "candles_to_trade": 1,
        "prediction_length": 4,
        "results_output_dir": "1h_4pred",
    }
})
RECALCULATION_AFTER_REPEATED_SIGNAL = CONFIG.get("recalc_after_repeated_signal", True)
CRYPTOCOMPARE_API_KEY = CONFIG.get("news", {}).get("cryptocompare_api_key", "")
NEWS_LIMIT = CONFIG.get("news", {}).get("limit", 100)
NEWS_TIMEOUT = CONFIG.get("news", {}).get("timeout", 20)
MAIN_COINS = CONFIG.get("main_coins", ["BTCUSDT", "ETHUSDT"])
API_KEY = CONFIG.get("api_key", "")
API_SECRET = CONFIG.get("api_secret", "")
BASE_URL = CONFIG.get("base_url", "https://api-demo.bybit.com")
SYMBOLS = CONFIG.get("symbols", [])
LEVERAGE_CONFIG = CONFIG.get("leverage", {"default": 30})

if not TIMEFRAME_CONFIG.get("LONG_TF", {}).get("use", True):
    logging.critical(
        "LONG_TF model is disabled. The bot cannot operate without it."
    )
    sys.exit(1)

matplotlib.use('Agg')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        RotatingFileHandler("bot.log", maxBytes=10**7, backupCount=5),
        logging.StreamHandler()
    ]
)


PREDICTION_MANAGER: PredictionManager | None = None

GLOBAL_SIGNALS_LONG_TF = {}

GLOBAL_PREDICT_SEMAPHORE = asyncio.Semaphore(16)

GLOBAL_SIGNALS_VERSION = 0
GLOBAL_SIGNALS_UPDATE_EVENT = asyncio.Event()

ALL_BOTS = []

def init_models_once():
    """Initialize PredictionManager and load the LONG_TF model once."""
    global PREDICTION_MANAGER

    long_config = TIMEFRAME_CONFIG["LONG_TF"]

    try:
        config_long_tf = {
            "seq_len": long_config.get("seq_len", 512),
            "results_output_dir": long_config.get("results_output_dir", "1h_4pred"),
            "model_name": long_config.get("model_name", "NeoQuasar/Kronos-mini"),
            "prediction_length": long_config.get("prediction_length", 1),
        }
        manager = PredictionManager(config_long_tf)
        manager.start()
        manager.ready.wait()
        if manager.model is None:
            raise RuntimeError("PredictionManager failed to initialize")

        PREDICTION_MANAGER = manager
        logging.info("LONG_TF model loaded successfully.")
        return True

    except Exception as e:
        logging.exception(f"Error initializing LONG_TF model: {e}")
        return False

async def fetch_market_data_for_symbols(symbols, tf, total_candles=10500) -> pd.DataFrame:
    """Load historical candles for the provided symbols."""
    async def gather_fetches():
        tasks = [
            asyncio.to_thread(
                fetch_data_with_metrics,
                None, None, BASE_URL, sym, tf, 1000, total_candles, 3, "data"
            )
            for sym in symbols
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = await gather_fetches()
    all_dfs = []

    for i, sym in enumerate(symbols):
        result = results[i]
        if isinstance(result, Exception):
            logging.error(
                f"Error loading data symbol={sym}, tf={tf}: {result}"
            )
        elif isinstance(result, pd.DataFrame) and not result.empty:
            result["item_id"] = sym
            all_dfs.append(result)

    if not all_dfs:
        return pd.DataFrame()

    df_big = pd.concat(all_dfs, ignore_index=True)

    if "timestamp" not in df_big.columns:
        logging.error(
            "Result dataframe lacks 'timestamp' column. Returning empty."
        )
        return pd.DataFrame()
    
    return df_big

async def get_signals_for_all_symbols(symbols):
    """Get LONG_TF model signals for all provided symbols."""
    global GLOBAL_SIGNALS_LONG_TF
    global GLOBAL_SIGNALS_VERSION

    long_config = TIMEFRAME_CONFIG["LONG_TF"]

    async def fetch_and_process_long():
        if PREDICTION_MANAGER is None:
            return {sym: "NEUTRAL" for sym in symbols}
        async with GLOBAL_PREDICT_SEMAPHORE:
            df = await fetch_market_data_for_symbols(symbols, long_config["interval"], total_candles=4000)
        if not df.empty:
            return await PREDICTION_MANAGER.predict(df)
        else:
            return {sym: "NEUTRAL" for sym in symbols}

    signals_long_tf = await fetch_and_process_long()

    GLOBAL_SIGNALS_LONG_TF = signals_long_tf

    GLOBAL_SIGNALS_VERSION += 1
    GLOBAL_SIGNALS_UPDATE_EVENT.set()
    logging.info(
        f"SIGNALS: new LONG_TF signals received, version={GLOBAL_SIGNALS_VERSION}"
    )
    signals_arr = ", ".join(f"{sym}: {sig}" for sym, sig in signals_long_tf.items())
    logging.info(f"SIGNALS_ARRAY: [{signals_arr}]")

    return True

class WebSocketManager:
    def __init__(self, ws_url, symbol, interval_str, accumulation=1):
        self.ws_url = ws_url
        self.symbol = symbol
        self.interval = timeframe_to_interval(interval_str)
        self.interval_str = interval_str
        self.websocket = None
        self.running = False
        self.bots = []
        self.last_candle_open_time = None
        self.loop = None
        self.candle_accumulation_threshold = accumulation
        self.confirmed_candles_counter = 0

    async def connect(self):
        max_retries = 100
        retry_delay = 2
        attempt = 0
        self.running = True

        while attempt < max_retries and self.running:
            attempt += 1
            try:
                logging.info(
                    f"WebSocketManager: Connecting to {self.ws_url} for {self.symbol} ({self.interval_str}), attempt {attempt}/{max_retries}"
                )
                async with websockets.connect(self.ws_url, open_timeout=20, close_timeout=10, ping_interval=20, ping_timeout=20) as websocket:
                    self.websocket = websocket
                    sub_msg = {"op": "subscribe", "args": [f"kline.{self.interval}.{self.symbol}"]}
                    await websocket.send(json.dumps(sub_msg))
                    logging.info(
                        f"WebSocketManager: Subscribed to {self.symbol} ({self.interval_str})"
                    )
                    await self.process_messages(websocket)

            except (websockets.exceptions.ConnectionClosed, websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK,  websockets.exceptions.InvalidURI, websockets.exceptions.InvalidHandshake, asyncio.TimeoutError) as e:
                logging.warning(
                    f"WebSocketManager: WebSocket ({self.symbol}, {self.interval_str}) closed or timed out: {type(e).__name__} - {e}, attempt {attempt}/{max_retries}"
                )
                if isinstance(e, websockets.exceptions.ConnectionClosedOK):
                     if not self.running: break
                if attempt < max_retries and self.running:
                    await asyncio.sleep(min(retry_delay, 30))
                    retry_delay *= 1.1
                elif self.running:
                    logging.critical(
                        f"WebSocketManager: Maximum connection attempts reached for {self.symbol} ({self.interval_str})"
                    )
                    break

            except Exception as e:
                logging.exception(
                    f"WebSocketManager: Unknown WebSocket error ({self.symbol}, {self.interval_str}): {e}"
                )
                if attempt < max_retries and self.running:
                    await asyncio.sleep(min(retry_delay, 30))
                    retry_delay *= 1.1
                elif self.running:
                    logging.critical(
                        f"WebSocketManager: Maximum connection attempts reached (unknown error) for {self.symbol} ({self.interval_str})"
                    )
                    break

            finally:
                 self.websocket = None

        logging.info(
            f"WebSocketManager ({self.symbol}, {self.interval_str}): Connection loop finished."
        )

    async def process_messages(self, websocket):

        async for msg in websocket:
            if not self.running:
                break
            try:
                data = json.loads(msg)

                if data.get("op") == "subscribe" and data.get("success"):
                    logging.info(
                        f"WebSocketManager: Subscription confirmed: {data.get('ret_msg')}"
                    )
                    continue
                if data.get("op") == "ping":
                     await websocket.send(json.dumps({"op": "pong", "req_id": data.get("req_id")}))
                     continue
                if data.get("op") == "pong":
                     continue

                if data.get('topic', '').startswith('kline') and 'data' in data:
                     for candle in data['data']:
                         confirm = candle.get('confirm', False)
                         start = candle.get('start', None)

                         if confirm and start is not None:
                             current_start_time = int(start)
                             if current_start_time != self.last_candle_open_time:
                                 self.last_candle_open_time = current_start_time
                                 self.confirmed_candles_counter += 1
                                 logging.info(
                                     f"WebSocketManager: Candle confirmed ({self.interval_str}) [{self.confirmed_candles_counter}/{self.candle_accumulation_threshold}], timestamp={current_start_time}"
                                 )
                                 
                                 if self.confirmed_candles_counter >= self.candle_accumulation_threshold:
                                     self.confirmed_candles_counter = 0
                                     await get_signals_for_all_symbols(SYMBOLS)
                                     await GLOBAL_SIGNALS_UPDATE_EVENT.wait()
                                     signals_version = GLOBAL_SIGNALS_VERSION
                                     GLOBAL_SIGNALS_UPDATE_EVENT.clear()

                                     trade_tasks = []
                                     for bot in self.bots:
                                         bot.signals_version = signals_version
                                         trade_tasks.append(asyncio.create_task(bot.trade()))

                                     logging.info(
                                         f"WebSocketManager: Trade tasks for {len(trade_tasks)} bots started (version={signals_version})."
                                     )

            except json.JSONDecodeError:
                logging.debug(
                    f"WebSocketManager: JSON decode error: {msg}"
                )
            except Exception as e:
                logging.exception(
                    f"WebSocketManager: Error processing message: {e}"
                )

    def register_bot(self, bot):
        self.bots.append(bot)
        logging.info(f"WebSocketManager: Registered bot {bot.name}")

    def start(self):
        def run_manager():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_until_complete(self.connect())
            finally:
                logging.info(
                    f"WebSocketManager ({self.symbol}, {self.interval_str}): Closing event loop."
                )

                tasks = asyncio.all_tasks(loop=self.loop)
                for task in tasks:
                     task.cancel()

                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.close()
                asyncio.set_event_loop(None)

        thread = Thread(target=run_manager, name=f"WebSocketManager-{self.symbol}-{self.interval_str}")
        thread.daemon = True
        thread.start()
        return thread

    def stop(self):
        logging.info(
            f"WebSocketManager ({self.symbol}, {self.interval_str}): Stop requested."
        )
        self.running = False

        if self.websocket and self.loop and self.loop.is_running():
             logging.info(
                 f"WebSocketManager ({self.symbol}, {self.interval_str}): Sending WebSocket close command."
             )
             future = asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
             try:
                future.result(timeout=5)
                logging.info(
                    f"WebSocketManager ({self.symbol}, {self.interval_str}): WebSocket closed successfully."
                )
             except asyncio.TimeoutError:
                logging.warning(
                    f"WebSocketManager ({self.symbol}, {self.interval_str}): Timeout while closing WebSocket."
                )
             except Exception as e:
                logging.error(
                    f"WebSocketManager ({self.symbol}, {self.interval_str}): Error closing WebSocket: {e}"
                )
        elif self.loop and self.loop.is_running():
             self.loop.call_soon_threadsafe(self.loop.stop)
             logging.info(
                 f"WebSocketManager ({self.symbol}, {self.interval_str}): Event loop stop command sent."
             )

class ModelBasedBot(Thread):
    def __init__(self, api_key, api_secret, base_url, symbol, leverage=30, news_check_enabled=True):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.symbol = symbol
        self.leverage = leverage
        self.news_check_enabled = news_check_enabled
        self.name = f"Bot-{self.symbol}"
        self.running = True

        try:
            self.equity = asyncio.run(asyncio.to_thread(get_account_balance, self.base_url, self.api_key, self.api_secret))
            self.starting_balance = self.equity
            logging.info(f"{self.name}: Initialized with balance: {self.equity}")
        except Exception as e:
            logging.critical(
                f"{self.name}: Failed to fetch balance during initialization: {e}"
            )
            self.equity = 0.0
            self.starting_balance = 0.0

        self.risk_factor = 0.05 if self.symbol in MAIN_COINS else 0.035
        self.drawdown_alert_threshold = 0.2
        self.drawdown_alert_triggered = False

        self.position_size = 0.0
        self.position_type = None

        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
        self.signals_log_dir = 'signals'
        os.makedirs(self.signals_log_dir, exist_ok=True)
        self.signals_file = os.path.join(self.signals_log_dir, f'signals_{self.symbol}.log')

        self.session = self._setup_requests_session()
        self.last_news_timestamp = None
        self.news_fetched = False

        self.default_config = {
            "STOP_LOSS_ATR_MULTIPLIER": 3,
            "TAKE_PROFIT_ATR_MULTIPLIER": 6,
            "TRAILING_STOP_PERCENT": 0.02,
            "USE_TRAILING_STOP": False,
            "MIN_TP_SL_PERCENT": 0.0,
        }

        self.signals_version = -1
        self.current_signal_version = -1

    def _setup_requests_session(self):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        return session

    async def initial_setup(self):
        try:
            await self.initial_news_fetch()

            lev_result = await asyncio.to_thread(
                set_leverage, self.base_url, self.api_key, self.api_secret, self.symbol, self.leverage
            )
            if lev_result.get('retCode') != 0:
                logging.warning(
                    f"{self.name}: Failed to set leverage {self.leverage}: {lev_result.get('retMsg')}"
                )

            margin_result = await asyncio.to_thread(
                switch_to_cross_margin, self.base_url, self.api_key, self.api_secret, self.symbol
            )
            if margin_result.get('retCode') != 0:
                 if "position mode is not modified" not in margin_result.get('retMsg', '').lower():
                      logging.warning(
                          f"{self.name}: Failed to switch to cross margin: {margin_result.get('retMsg')}"
                      )

            logging.info(f"{self.name}: Initial setup completed.")

        except Exception as e:
            logging.exception(f"{self.name}: Error during initial_setup: {e}")

    async def initial_news_fetch(self):
        """Fetch news on startup to initialize ``last_news_timestamp``."""
        if not self.news_check_enabled or self.news_fetched:
            return
        try:
            logging.debug(f"{self.name}: Performing initial news fetch...")
            news_articles = await self.get_news(initial=True)
            if news_articles:
                valid_timestamps = [a['PUBLISHED_ON'] for a in news_articles if 'PUBLISHED_ON' in a and isinstance(a['PUBLISHED_ON'], int)]
                if valid_timestamps:
                    self.last_news_timestamp = max(valid_timestamps)
                    logging.info(
                        f"{self.name}: Initial news fetch completed. last_news_timestamp={self.last_news_timestamp}"
                    )
                else:
                     logging.info(
                         f"{self.name}: No valid timestamps found in initial news."
                     )
            else:
                logging.info(f"{self.name}: No initial news found.")
            self.news_fetched = True
        except Exception as e:
            logging.exception(
                f"{self.name}: Error during initial news fetch: {e}"
            )

    async def get_news(self, initial=False):
        """Fetch and filter news from CryptoCompare."""
        if not self.news_check_enabled:
            return []

        cleaned_symbol = re.sub(r'\d+', '', self.symbol.replace("USDT", "")).upper()
        url = "https://data-api.cryptocompare.com/news/v1/article/list"
        params = {'limit': NEWS_LIMIT, 'categories': cleaned_symbol, 'api_key': CRYPTOCOMPARE_API_KEY}
        headers = {"Accept": "application/json"}

        try:
            response = await asyncio.to_thread(self.session.get, url, headers=headers, params=params, timeout=NEWS_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            if 'Err' in data and data['Err']:
                logging.warning(
                    f"{self.name}: CryptoCompare API error: {data['Err']}"
                )
                return []

            articles = data.get('Data', [])
            if not articles:
                return []

            new_articles = []
            current_max_ts = self.last_news_timestamp or 0

            for article in articles:
                published_on = article.get('PUBLISHED_ON', 0)
                if not isinstance(published_on, int) or published_on == 0:
                    continue

                if initial or published_on > current_max_ts:
                    new_articles.append(article)

            if not initial and new_articles:
                latest_ts_in_batch = max(a['PUBLISHED_ON'] for a in new_articles)
                if latest_ts_in_batch > current_max_ts:
                     self.last_news_timestamp = latest_ts_in_batch
                     logging.debug(
                         f"{self.name}: Updated last_news_timestamp={self.last_news_timestamp}"
                     )

            return new_articles

        except requests.exceptions.RequestException as e:
            logging.error(
                f"{self.name}: Network error while fetching news: {e}"
            )
            return []
        except Exception as e:
            logging.exception(
                f"{self.name}: Unknown error while fetching news: {e}"
            )
            return []

    def analyze_sentiment(self, articles):
        """Analyze news sentiment with exponential time decay."""
        if not articles:
            return 'NEUTRAL'

        try:
            current_time = time.time()
            max_age_seconds = 24 * 3600
            half_life_seconds = 4 * 3600
            lambda_decay = math.log(2) / half_life_seconds

            sentiment_weights = {'POSITIVE': 0.0, 'NEGATIVE': 0.0, 'NEUTRAL': 0.0}

            for article in articles:
                sentiment = article.get('SENTIMENT', 'NEUTRAL').upper()
                published_on = article.get('PUBLISHED_ON', 0)

                if sentiment not in sentiment_weights or not published_on:
                    continue

                age_seconds = current_time - published_on
                if age_seconds < 0 or age_seconds > max_age_seconds:
                    continue

                weight = math.exp(-lambda_decay * age_seconds)
                sentiment_weights[sentiment] += weight

            if not any(sentiment_weights.values()):
                 return 'NEUTRAL'

            threshold_ratio = 1.5
            dominant_sentiment = 'NEUTRAL'

            sorted_sentiments = sorted(sentiment_weights.items(), key=lambda item: item[1], reverse=True)

            if sorted_sentiments[0][1] > 0:
                 if len(sorted_sentiments) == 1 or sorted_sentiments[1][1] == 0:
                      dominant_sentiment = sorted_sentiments[0][0]
                 elif sorted_sentiments[0][1] >= sorted_sentiments[1][1] * threshold_ratio:
                      dominant_sentiment = sorted_sentiments[0][0]

            return dominant_sentiment

        except Exception as e:
            logging.exception(
                f"{self.name}: Error analyzing sentiment: {e}"
            )
            return 'NEUTRAL'

    async def check_drawdown(self):
        """Check drawdown, close the position if threshold exceeded and stop trading."""
        try:
            self.equity = await asyncio.to_thread(get_account_balance, self.base_url, self.api_key, self.api_secret)
            current_drawdown = (self.starting_balance - self.equity) / self.starting_balance if self.starting_balance > 0 else 0

            if current_drawdown >= self.drawdown_alert_threshold:
                if not self.drawdown_alert_triggered:
                    logging.warning(
                        f"{self.name}: Drawdown threshold reached ({current_drawdown:.2%})! Balance: {self.equity:.2f}"
                    )
                    self.drawdown_alert_triggered = True

                    if self.position_type:
                        await self.close_position(reason="Max drawdown reached, terminating trading")

                    logging.warning(
                        f"{self.name}: Maximum drawdown reached. Trading is not recommended!"
                    )

        except Exception as e:
            logging.exception(f"{self.name}: Error checking drawdown: {e}")

    def calculate_position_size(self, current_price):
        """Calculate position size considering risk, leverage and exchange limits."""
        try:
            if self.equity <= 0 or current_price <= 0:
                logging.warning(
                    f"{self.name}: Cannot calculate position size (equity={self.equity}, price={current_price})"
                )
                return 0.0

            risk_amount_usd = self.risk_factor * self.equity
            position_size_usd = risk_amount_usd * self.leverage
            position_size_coin = position_size_usd / current_price

            min_order_qty = get_min_order_size(self.base_url, self.symbol)
            max_order_qty = get_max_qty(self.base_url, self.symbol)

            if position_size_coin < min_order_qty:
                 logging.warning(
                     f"{self.name}: Calculated size ({position_size_coin:.8f}) below minimum ({min_order_qty}). Using minimum."
                 )
                 position_size_coin = min_order_qty
            elif position_size_coin > max_order_qty:
                 logging.warning(
                     f"{self.name}: Calculated size ({position_size_coin:.8f}) above maximum ({max_order_qty}). Using maximum."
                 )
                 position_size_coin = max_order_qty

            step_size = min_order_qty
            precision = abs(int(math.log10(step_size))) if step_size > 0 else 8
            position_size_coin = math.floor(position_size_coin / step_size) * step_size
            position_size_coin = round(position_size_coin, precision)

            if position_size_coin < min_order_qty:
                 logging.warning(
                     f"{self.name}: Position size after rounding ({position_size_coin:.8f}) below minimum. Returning 0."
                 )
                 return 0.0

            logging.info(
                f"{self.name}: Calculated position size: {position_size_coin:.8f} {self.symbol.replace('USDT', '')}"
            )
            return position_size_coin

        except Exception as e:
            logging.exception(
                f"{self.name}: Error calculating position size: {e}"
            )
            return 0.0

    async def trade(self):
        if self.signals_version <= self.current_signal_version:
            return

        await self.check_drawdown()
        if not self.running:
            return

        self.current_signal_version = self.signals_version
        logging.info(
            f"{self.name}: Starting processing of signals version {self.current_signal_version}"
        )

        try:
            signal = GLOBAL_SIGNALS_LONG_TF.get(self.symbol, "NEUTRAL")
            logging.info(
                f"{self.name}: Received LONG_TF signal ({TIMEFRAME_CONFIG['LONG_TF']['interval']}) => {signal} (v{self.current_signal_version})"
            )

            sentiment = 'NEUTRAL'
            if self.news_check_enabled:
                news_articles = await self.get_news()
                sentiment = self.analyze_sentiment(news_articles)
                logging.info(f"{self.name}: News sentiment => {sentiment}")

            position_data = await asyncio.to_thread(get_open_position, self.base_url, self.api_key, self.api_secret, self.symbol)
            current_position = None
            if isinstance(position_data, list) and len(position_data) > 0:
                current_position = position_data[0]
            elif isinstance(position_data, dict) and position_data.get('size') and float(position_data['size']) > 0:
                current_position = position_data

            if current_position:
                self.position_type = current_position['side'].lower()
                self.position_size = float(current_position['size'])
                entry_price = float(current_position.get('avgPrice') or current_position.get('entryPrice', 0))
                logging.info(
                    f"{self.name}: Active position: {self.position_type.upper()} | Size: {self.position_size} | Entry: {entry_price:.4f}"
                )
            else:
                if self.position_type:
                    logging.info(f"{self.name}: Position no longer active.")
                self.position_type = None
                self.position_size = 0.0

            market_data_tf = TIMEFRAME_CONFIG["LONG_TF"]["interval"]
            df_price = await asyncio.to_thread(fetch_data_with_metrics, None, None, None, self.symbol, market_data_tf, 2, 2, 3, self.data_dir)

            if df_price.empty or 'close' not in df_price.columns:
                logging.warning(
                    f"{self.name}: Failed to get current price ({market_data_tf}). Skipping trading logic."
                )
                return

            current_price = df_price['close'].iloc[-1]
            if current_price <= 0:
                logging.warning(
                    f"{self.name}: Invalid price received ({current_price}). Skipping trading logic."
                )
                return

            action_taken = False

            if self.position_type is None:
                self.equity = await asyncio.to_thread(get_account_balance, self.base_url, self.api_key, self.api_secret)
                if signal == "AGREE_LONG":
                    if not self.news_check_enabled or sentiment == 'POSITIVE':
                        logging.info(
                            f"{self.name}: Signal AGREE_LONG and suitable sentiment ({sentiment}). Attempting to open LONG."
                        )
                        qty = self.calculate_position_size(current_price)
                        if qty > 0:
                            await self.open_position('Buy', qty, current_price)
                            action_taken = True
                    else:
                        logging.info(
                            f"{self.name}: Signal AGREE_LONG but sentiment ({sentiment}) does not allow opening LONG."
                        )
                elif signal == "AGREE_SHORT":
                    if not self.news_check_enabled or sentiment == 'NEGATIVE':
                        logging.info(
                            f"{self.name}: Signal AGREE_SHORT and suitable sentiment ({sentiment}). Attempting to open SHORT."
                        )
                        qty = self.calculate_position_size(current_price)
                        if qty > 0:
                            await self.open_position('Sell', qty, current_price)
                            action_taken = True
                    else:
                        logging.info(
                            f"{self.name}: Signal AGREE_SHORT but sentiment ({sentiment}) does not allow opening SHORT."
                        )

            elif self.position_type == 'buy':
                if signal == "AGREE_LONG":
                    if RECALCULATION_AFTER_REPEATED_SIGNAL:
                        logging.info(
                            f"{self.name}: RECALCULATION_AFTER_REPEATED_SIGNAL enabled. Recalculating SL for current LONG position."
                        )
                        await self.set_stop_loss_take_profit('buy', current_price, self.position_size)
                        action_taken = True
                    else:
                        logging.info(
                            f"{self.name}: Signal AGREE_LONG confirms holding LONG."
                        )
                elif signal == "AGREE_SHORT":
                    logging.info(f"{self.name}: Signal AGREE_SHORT. Closing LONG.")
                    await self.close_position(reason="Signal reverse to AGREE_SHORT")
                    action_taken = True

                    self.equity = await asyncio.to_thread(get_account_balance, self.base_url, self.api_key, self.api_secret)
                    if not self.news_check_enabled or sentiment == 'NEGATIVE':
                        logging.info(
                            f"{self.name}: Attempting reversal to SHORT after closing LONG."
                        )
                        qty = self.calculate_position_size(current_price)
                        if qty > 0:
                            await self.open_position('Sell', qty, current_price)
                    else:
                        logging.info(
                            f"{self.name}: Sentiment ({sentiment}) does not allow reversal to SHORT."
                        )
                elif signal == "DISAGREE_SHORT" or signal == "NEUTRAL" or signal == "DISAGREE_LONG":
                    logging.info(
                        f"{self.name}: Signal {signal} indicates weakening/no LONG. Closing LONG."
                    )
                    await self.close_position(reason=f"Close LONG on {signal} signal")
                    action_taken = True

            elif self.position_type == 'sell':
                if signal == "AGREE_SHORT":
                    if RECALCULATION_AFTER_REPEATED_SIGNAL:
                        logging.info(
                            f"{self.name}: RECALCULATION_AFTER_REPEATED_SIGNAL enabled. Recalculating SL for current SHORT position."
                        )
                        await self.set_stop_loss_take_profit('sell', current_price, self.position_size)
                        action_taken = True
                    else:
                        logging.info(
                            f"{self.name}: Signal AGREE_SHORT confirms holding SHORT."
                        )
                elif signal == "AGREE_LONG":
                    logging.info(f"{self.name}: Signal AGREE_LONG. Closing SHORT.")
                    await self.close_position(reason="Signal reverse to AGREE_LONG")
                    action_taken = True

                    self.equity = await asyncio.to_thread(get_account_balance, self.base_url, self.api_key, self.api_secret)
                    if not self.news_check_enabled or sentiment == 'POSITIVE':
                        logging.info(
                            f"{self.name}: Attempting reversal to LONG after closing SHORT."
                        )
                        qty = self.calculate_position_size(current_price)
                        if qty > 0:
                            await self.open_position('Buy', qty, current_price)
                    else:
                        logging.info(
                            f"{self.name}: Sentiment ({sentiment}) does not allow reversal to LONG."
                        )
                elif signal == "DISAGREE_LONG" or signal == "NEUTRAL" or signal == "DISAGREE_SHORT":
                    logging.info(
                        f"{self.name}: Signal {signal} indicates weakening/no SHORT. Closing SHORT."
                    )
                    await self.close_position(reason=f"Close SHORT on {signal} signal")
                    action_taken = True

            if not action_taken:
                logging.info(
                    f"{self.name}: Current signal ({signal}) and position ({self.position_type}) require no action."
                )

        except Exception as e:
            logging.exception(f"{self.name}: Critical error in trade method: {e}")
        finally:
            logging.debug(
                f"{self.name}: Finished processing signals version {self.current_signal_version}"
            )

    async def open_position(self, side, qty, current_price):
        """Open a position and set initial SL/TP levels."""
        if qty <= 0:
            logging.warning(
                f"{self.name}: Attempt to open a position with non-positive quantity ({qty})."
            )
            return

        try:
            log_msg = f"ACTION: Open {side.upper()} | Price: {current_price:.4f} | Qty: {qty}"
            self.log_signal(log_msg)

            order_response = await asyncio.to_thread(
                open_position_api,
                self.base_url, self.api_key, self.api_secret,
                self.symbol, side, qty, self.default_config
            )

            if order_response.get('retCode') == 0 and order_response.get('result', {}).get('orderId'):
                order_id = order_response['result']['orderId']
                logging.info(
                    f"{self.name}: {side.upper()} order {qty} {self.symbol} placed successfully (ID: {order_id})."
                )
                self.position_type = side.lower()
                self.position_size = qty

                await self.set_stop_loss_take_profit(side, current_price, qty)
            else:
                error_msg = order_response.get('retMsg', 'Unknown API error')
                logging.error(
                    f"{self.name}: Failed to place {side.upper()} order: {error_msg} (Code: {order_response.get('retCode')})"
                )

                self.position_type = None
                self.position_size = 0.0

        except Exception as e:
            logging.exception(f"{self.name}: Error opening position {side.upper()}: {e}")
            self.position_type = None
            self.position_size = 0.0

    async def set_stop_loss_take_profit(self, side, price, qty):
        try:
            atr_tf = TIMEFRAME_CONFIG['LONG_TF']['interval']
            df_atr = await asyncio.to_thread(fetch_data_with_metrics, self.api_key, self.api_secret, self.base_url, self.symbol, atr_tf, 100, 100, 3, self.data_dir)

            if df_atr.empty:
                logging.warning(
                    f"{self.name}: Failed to fetch data for ATR calculation ({atr_tf}). SL/TP not set."
                )
                return

            atr_series = calculate_atr(df_atr)
            if atr_series is None or atr_series.empty or pd.isna(atr_series.iloc[-1]):
                logging.warning(
                    f"{self.name}: Failed to calculate ATR. SL/TP not set."
                )
                return

            last_atr = atr_series.iloc[-1]
            if last_atr <= 0:
                logging.warning(
                    f"{self.name}: Calculated invalid ATR ({last_atr}). SL/TP not set."
                )
                return

            sl_multiplier = self.default_config["STOP_LOSS_ATR_MULTIPLIER"]
            tp_multiplier = self.default_config["TAKE_PROFIT_ATR_MULTIPLIER"]
            min_pct = self.default_config.get("MIN_TP_SL_PERCENT", 0)
            sl_price = 0.0
            tp_price = 0.0

            if side.lower() == 'buy':
                sl_price = price - (sl_multiplier * last_atr)
                tp_price = price + (tp_multiplier * last_atr)
            else:
                sl_price = price + (sl_multiplier * last_atr)
                tp_price = price - (tp_multiplier * last_atr)

                if tp_price <= 0: tp_price = price * 0.01

            if min_pct > 0:
                min_diff = price * min_pct
                adjusted = False
                if abs(tp_price - price) < min_diff:
                    adjusted = True
                    if side.lower() == 'buy':
                        tp_price = price + min_diff
                    else:
                        tp_price = max(price - min_diff, 0)
                if abs(sl_price - price) < min_diff:
                    adjusted = True
                    if side.lower() == 'buy':
                        sl_price = max(price - min_diff, 0)
                    else:
                        sl_price = price + min_diff
                if adjusted:
                    logging.info(
                        f"{self.name}: Adjusting TP/SL according to minimum threshold {min_pct*100:.0f}%: SL={sl_price:.4f}, TP={tp_price:.4f}"
                    )

            trailing_stop_distance = 0.0
            if self.default_config["USE_TRAILING_STOP"]:
                trailing_stop_distance = self.default_config["TRAILING_STOP_PERCENT"] * price

            logging.info(
                f"{self.name}: Calculated levels: SL={sl_price:.4f}, TP={tp_price:.4f}, Trailing={trailing_stop_distance:.4f} (ATR={last_atr:.4f})"
            )

            stop_response = await asyncio.to_thread(set_trading_stop, self.base_url, self.api_key, self.api_secret, self.symbol, sl_price, tp_price, trailing_stop_distance, 'Full', self.default_config)

            if stop_response.get('retCode') == 0:
                logging.info(f"{self.name}: SL/TP set request sent successfully.")
            else:
                error_msg = stop_response.get('retMsg', 'Unknown API error')
                ignore_errors = ["stop loss price is not valid", "take profit price is not valid", "order quantity not match position side"]
                should_log_error = True
                for err in ignore_errors:
                    if err in error_msg.lower():
                        logging.warning(
                            f"{self.name}: Could not set SL/TP (ignored error): {error_msg}"
                        )
                        should_log_error = False
                        break
                     
                if should_log_error:
                    logging.error(
                        f"{self.name}: Error setting SL/TP: {error_msg} (Code: {stop_response.get('retCode')})"
                    )

        except Exception as e:
            logging.exception(f"{self.name}: Error while setting SL/TP: {e}")

    async def close_position(self, reason="N/A", max_retries=3):
        """Close the current open position."""
        if not self.position_type:
            logging.info(f"{self.name}: No open position to close.")
            return

        current_pos_type = self.position_type
        log_msg = f"ACTION: Close {current_pos_type.upper()} | Reason: {reason}"
        self.log_signal(log_msg)

        attempt = 0
        while attempt < max_retries:
            attempt += 1
            logging.info(
                f"{self.name}: Attempt {attempt}/{max_retries} to close position {current_pos_type.upper()}..."
            )
            try:
                close_response = await asyncio.to_thread(
                    close_positions, self.base_url, self.api_key, self.api_secret, self.symbol
                )

                if close_response.get('retCode') == 0:
                    logging.info(
                        f"{self.name}: Close order for {current_pos_type.upper()} placed successfully."
                    )
                    await asyncio.sleep(2)
                    position_check = await asyncio.to_thread(get_open_position, self.base_url, self.api_key, self.api_secret, self.symbol)

                    is_closed = True
                    if isinstance(position_check, list) and len(position_check) > 0:
                        is_closed = False
                    elif isinstance(position_check, dict) and position_check.get('size') and float(position_check['size']) > 0:
                        is_closed = False

                    if is_closed:
                        logging.info(
                            f"{self.name}: Position {current_pos_type.upper()} closed successfully (confirmed)."
                        )
                        self.position_type = None
                        self.position_size = 0.0
                        return
                    else:
                        logging.warning(
                            f"{self.name}: Position {current_pos_type.upper()} still active after close attempt {attempt}."
                        )
                        if attempt >= max_retries:
                             logging.error(
                                 f"{self.name}: Could not confirm closing {current_pos_type.upper()} position after {max_retries} attempts."
                             )
                             return
                else:
                    error_msg = close_response.get('retMsg', 'Unknown API error')
                    if "position idx not match position side" in error_msg or "order already cancelled or filled" in error_msg:
                        logging.warning(
                            f"{self.name}: Attempted to close position but it is already closed or order filled: {error_msg}"
                        )
                        self.position_type = None
                        self.position_size = 0.0
                        return
                    else:
                        logging.error(
                            f"{self.name}: API error while closing {current_pos_type.upper()} position (attempt {attempt}): {error_msg}"
                        )

            except Exception as e:
                logging.exception(
                    f"{self.name}: Exception while closing {current_pos_type.upper()} position (attempt {attempt}): {e}"
                )

            if attempt < max_retries:
                await asyncio.sleep(attempt * 2)

        logging.error(
            f"{self.name}: Failed to close {current_pos_type.upper()} position after {max_retries} attempts."
        )

    def log_signal(self, message):
        """Write a message to the signal log file and main log."""
        try:
            full_message = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {self.name} - {message}"
            with open(self.signals_file, 'a', encoding='utf-8') as f:
                f.write(full_message + '\n')
            logging.info(message)
        except Exception as e:
            logging.error(f"{self.name}: Failed to write to signal log: {e}")

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.initial_setup())
            self.loop.run_forever()
        except asyncio.CancelledError:
             logging.info(f"{self.name}: Main loop cancelled.")
        except Exception as e:
            logging.critical(f"{self.name}: Critical error in main loop: {e}", exc_info=True)
        finally:
            logging.info(f"{self.name}: Beginning bot shutdown...")

            tasks = asyncio.all_tasks(loop=self.loop)
            for task in tasks:
                 if not task.done(): task.cancel()

            if self.loop.is_running():
                 self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                 self.loop.stop()
                 self.loop.close()
            logging.info(f"{self.name}: Bot stopped.")
            asyncio.set_event_loop(None)

    def stop(self):
         logging.info(f"{self.name}: Stop requested.")
         self.running = False
         if self.loop and self.loop.is_running():
              self.loop.call_soon_threadsafe(self.loop.stop)

if __name__ == "__main__":
    if not init_models_once():
        logging.critical("Failed to load LONG_TF model. Startup aborted.")
        sys.exit(1)

    logging.info("Fetching initial signals...")
    try:
        asyncio.run(get_signals_for_all_symbols(SYMBOLS))
        time.sleep(2)
    except Exception:
        logging.exception("Error while obtaining initial signals.")

    trading_interval = TIMEFRAME_CONFIG["LONG_TF"].get("interval", "1h")
    accumulation = TIMEFRAME_CONFIG["LONG_TF"].get("candles_to_trade", 4)

    ws_manager = WebSocketManager(
        ws_url='wss://stream.bybit.com/v5/public/linear',
        symbol='BTCUSDT',
        interval_str=trading_interval,
        accumulation=accumulation
    )

    ALL_BOTS = []
    enable_news_check = False

    for symbol in SYMBOLS:
        logging.info(f"Creating bot for {symbol}...")
        leverage = LEVERAGE_CONFIG.get(symbol, LEVERAGE_CONFIG.get("default", 30))
        bot = ModelBasedBot(
            API_KEY,
            API_SECRET,
            BASE_URL,
            symbol,
            leverage=leverage,
            news_check_enabled=enable_news_check,
        )
        ws_manager.register_bot(bot)
        ALL_BOTS.append(bot)

    logging.info("Starting WebSocket manager...")
    ws_thread = ws_manager.start()

    logging.info("Starting bot threads...")
    for bot in ALL_BOTS:
        bot.start()

    try:
        ws_thread.join()
        logging.info("WebSocket manager finished.")
        for bot in ALL_BOTS:
            bot.join()
            logging.info(f"Bot {bot.name} finished.")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Stopping...")
    except Exception as e:
        logging.critical(f"Unhandled error in main thread: {e}", exc_info=True)
    finally:
        logging.info("Stopping WebSocket manager...")
        ws_manager.stop()
        logging.info("Stopping all bots...")
        for bot in ALL_BOTS:
            bot.stop()

        if PREDICTION_MANAGER:
            PREDICTION_MANAGER.stop()

        shutdown_timeout = 35
        ws_thread.join(timeout=shutdown_timeout)
        if ws_thread.is_alive():
             logging.warning("WebSocket thread did not finish within timeout.")

        for bot in ALL_BOTS:
             bot.join(timeout=shutdown_timeout / len(ALL_BOTS) if ALL_BOTS else shutdown_timeout)
             if bot.is_alive():
                  logging.warning(f"Bot thread {bot.name} did not finish within timeout.")

        logging.info("Program finished.")
