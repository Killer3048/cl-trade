import ccxt
import hashlib
import hmac
import logging
import math
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from functools import wraps
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_metric_session = requests.Session()
_metric_adapter = HTTPAdapter(
    max_retries=Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
)
_metric_session.mount('https://', _metric_adapter)
_metric_session.mount('http://', _metric_adapter)


class APIRequestError(Exception):
    pass

def generate_signature(params, api_secret):
    ordered_params = sorted(params.items())
    query_string = '&'.join(f"{key}={value}" for key, value in ordered_params if value is not None)
    signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

def retry_on_failure(max_attempts=5, delay=2, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except APIRequestError as e:
                    attempts += 1
                    logging.error(f"APIRequestError in {func.__name__}: {e}. Attempt {attempts}/{max_attempts}")
                    if "not modified" in str(e).lower():
                        logging.warning(f"{func.__name__}: Специфическая ошибка '{e}', пропускаем дальнейшие попытки.")
                        break
                except Exception as e:
                    attempts += 1
                    logging.error(f"Unexpected error in {func.__name__}: {e}. Attempt {attempts}/{max_attempts}")
                time.sleep(current_delay)
                current_delay *= backoff
            logging.critical(f"Function {func.__name__} failed after {max_attempts} attempts")
            raise APIRequestError(f"Failed to execute function {func.__name__} after {max_attempts} attempts")
        return wrapper
    return decorator

def retry_async(max_attempts=5, delay=2, backoff=2):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except APIRequestError as e:
                    attempts += 1
                    logging.error(f"APIRequestError in {func.__name__}: {e}. Attempt {attempts}/{max_attempts}")
                    if "not modified" in str(e).lower():
                        logging.warning(f"{func.__name__}: Специфическая ошибка '{e}', пропускаем дальнейшие попытки.")
                        break
                except Exception as e:
                    attempts += 1
                    logging.error(f"Unexpected error in {func.__name__}: {e}. Attempt {attempts}/{max_attempts}")
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            logging.critical(f"Async function {func.__name__} failed after {max_attempts} attempts")
            raise APIRequestError(f"Failed to execute async function {func.__name__} after {max_attempts} attempts")
        return wrapper
    return decorator

@retry_on_failure()
def create_symbol_dir(data_dir, symbol):
    symbol_dir = os.path.join(data_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    return symbol_dir

@retry_on_failure(max_attempts=10, delay=3, backoff=2)
def get_open_position(base_url, api_key, api_secret, symbol):
    """
    Получение открытой позиции.
    Возвращает словарь с данными позиции, если позиция существует,
    или None, если открытых позиций не найдено.
    
    Дополнительно, если поле 'unrealisedPnl' присутствует,
    приводим его к типу float для удобства дальнейших проверок.
    """
    try:
        endpoint = '/v5/position/list'
        params = {
            'category': 'linear',
            'symbol': symbol,
            'api_key': api_key,
            'timestamp': int(time.time() * 1000)
        }
        params['sign'] = generate_signature(params, api_secret)
        response = requests.get(base_url + endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('retCode') != 0:
            raise APIRequestError(f"API Error: {data.get('retMsg')}")
        
        position_list = data.get('result', {}).get('list', [])
        
        for position in position_list:
            if position.get('symbol') == symbol and float(position.get('size', '0')) > 0:
                if 'unrealisedPnl' in position:
                    try:
                        position['unrealisedPnl'] = float(position['unrealisedPnl'])
                    except ValueError:
                        position['unrealisedPnl'] = 0.0
                logging.info(f"Получена открытая позиция для {symbol}: {position}")
                return position
        
        logging.info(f"Открытых позиций для {symbol} не найдено.")
        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error in get_open_position: {e} for {symbol}")
        raise APIRequestError(f"Request error: {e}")
    except (ValueError, KeyError, TypeError) as e:
        logging.error(f"Error processing response in get_open_position: {e}")
        raise APIRequestError(f"Error processing response: {e}")


@retry_on_failure()
def close_positions(base_url, api_key, api_secret, symbol):
    try:
        open_position = get_open_position(base_url, api_key, api_secret, symbol)
        if open_position:
            side = 'Sell' if open_position['side'] == 'Buy' else 'Buy'
            qty = open_position['size']
            response = place_order(base_url, api_key, api_secret, symbol, side, qty)
            logging.info(f"Закрыта позиция для {symbol}: {response}")
            return {'retCode': 0, 'retMsg': 'Position closed'}
        else:
            logging.info(f"Нет открытых позиций для закрытия по символу: {symbol}")
            return {'retCode': 0, 'retMsg': 'No open positions'}
    except APIRequestError as e:
        logging.error(f"API error while closing positions for {symbol}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error closing positions for {symbol}: {e}")
        raise

@retry_on_failure()
def place_order(base_url, api_key, api_secret, symbol, side, qty, order_type="Market", time_in_force="IOC"):
    endpoint = '/v5/order/create'
    params = {
        'category': 'linear',
        'symbol': symbol,
        'side': side,
        'orderType': order_type,
        'qty': str(qty),
        'timeInForce': time_in_force,
        'api_key': api_key,
        'timestamp': int(time.time() * 1000)
    }
    params['sign'] = generate_signature(params, api_secret)
    try:
        response = requests.post(base_url + endpoint, json=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result.get('retCode') != 0:
            raise APIRequestError(f"API error: {result.get('retMsg')}")

        return result
    except requests.exceptions.Timeout:
        logging.error(f"Timeout error in place_order for {symbol}")
        raise APIRequestError("Timeout during request")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error in place_order: {e}")
        raise APIRequestError(f"Request error: {e}")

@retry_on_failure()
def switch_to_cross_margin(base_url, api_key, api_secret, symbol):
    try:
        endpoint = '/v5/position/switch-mode'
        params = {
            'category': 'linear',
            'symbol': symbol,
            'mode': 0,
            'api_key': api_key,
            'timestamp': int(time.time() * 1000)
        }
        params['sign'] = generate_signature(params, api_secret)
        response = requests.post(base_url + endpoint, json=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result.get('retCode') != 0:
            ret_msg = result.get('retMsg', '').lower()
            if "position mode is not modified" in ret_msg:
                return result
            else:
                raise APIRequestError(f"API Error: {result.get('retMsg')}")
        logging.info(f"Режим маржи переключен для {symbol}: {result}")
        return result
    except requests.exceptions.Timeout:
        logging.error(f"Timeout error in switch_to_cross_margin for {symbol}")
        raise APIRequestError("Timeout during request")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error in switch_to_cross_margin: {e}")
        raise APIRequestError(f"Request error: {e}")
    except Exception as e:
        logging.error(f"Error in switch_to_cross_margin for {symbol}: {e}")
        raise APIRequestError(f"Error: {e}")

@retry_on_failure()
def set_leverage(base_url, api_key, api_secret, symbol, desired_leverage, default_leverage=30):
    try:
        if symbol in ["TONUSDT", "TRXUSDT"]:
            desired_leverage = 20

        endpoint = '/v5/position/set-leverage'
        params = {
            'category': 'linear',
            'symbol': symbol,
            'buyLeverage': str(desired_leverage),
            'sellLeverage': str(desired_leverage),
            'api_key': api_key,
            'timestamp': int(time.time() * 1000)
        }
        params['sign'] = generate_signature(params, api_secret)
        response = requests.post(base_url + endpoint, json=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result.get('retCode') != 0:
            ret_msg = result.get('retMsg', '').lower()
            if "leverage not modified" in ret_msg:
                return result
            else:
                raise APIRequestError(f"API Error: {result.get('retMsg')}")
        logging.info(f"Плечо установлено для {symbol}: {result}")
        return result
    except requests.exceptions.Timeout:
        logging.error(f"Timeout error in set_leverage for {symbol}")
        raise APIRequestError("Timeout during request")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error in set_leverage: {e}")
        raise APIRequestError(f"Request error: {e}")
    except Exception as e:
        logging.error(f"Error setting leverage for {symbol}: {e}")
        raise APIRequestError(f"Error setting leverage: {e}")

def create_bybit_client(api_key, api_secret, base_url):
    exchange = ccxt.bybit({
        'urls': {
            'api': {
                'public': 'https://api.bybit.com',
                'private': 'https://api.bybit.com',
            }
        }
    })
    return exchange

def timeframe_to_interval(timeframe):
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "1d": "D",
    }
    return mapping.get(timeframe, "1")

def timeframe_to_milliseconds(timeframe):
    mapping = {
        '1m': 60 * 1000,
        '3m': 3 * 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '2h': 2 * 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000
    }
    return mapping.get(timeframe, 60 * 1000)

def fetch_ohlcv_sync(exchange, symbol, timeframe, since, limit):
    """
    Синхронный вызов для получения OHLCV данных.
    """
    return exchange.fetch_ohlcv(symbol, timeframe, since, limit)

async def fetch_ohlcv_data_async(exchange, symbol, timeframe, limit, total_candles, retries, executor):
    """
    Асинхронная функция для получения OHLCV данных.
    Она вызывает синхронную функцию fetch_ohlcv_sync через run_in_executor.
    """
    interval = timeframe_to_milliseconds(timeframe)
    current_timestamp = exchange.milliseconds()
    since = current_timestamp - total_candles * interval
    all_ohlcv = []
    last_timestamp = None
    loop = asyncio.get_running_loop()

    while len(all_ohlcv) < total_candles and since < current_timestamp:
        data_fetched = False
        for attempt in range(retries):
            try:
                ohlcv = await loop.run_in_executor(
                    executor, fetch_ohlcv_sync, exchange, symbol, timeframe, since, limit
                )
                if not ohlcv:
                    logging.warning(f"No data for {symbol} on attempt {attempt + 1}")
                if last_timestamp is not None and ohlcv[0][0] <= last_timestamp:
                    break
                all_ohlcv.extend(ohlcv)
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + 1
                data_fetched = True
                await asyncio.sleep(0.005)
                break
            except Exception as e:
                logging.error(f"Error fetching data on attempt {attempt + 1}/{retries}: {e} for {symbol}")
                await asyncio.sleep(0.005)
        if not data_fetched:
            since += interval

    all_ohlcv = all_ohlcv[:total_candles]

    if not all_ohlcv:
        logging.error(f"No data for {symbol} for timeframe {timeframe}")
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
    df.set_index('timestamp', inplace=True)
    df = df.iloc[:-1]
    return df

@retry_on_failure()
def fetch_data(api_key, api_secret, base_url, symbol, timeframe='5m', limit=1000, total_candles=2000, retries=5, data_dir="data"):
    """
    Синхронная функция для получения данных.
    Внутри она запускает асинхронную функцию с asyncio.run().
    """
    exchange = create_bybit_client(api_key, api_secret, base_url)
    executor = ThreadPoolExecutor(max_workers=5)

    try:
        ohlcv_df = asyncio.run(
            fetch_ohlcv_data_async(exchange, symbol, timeframe, limit, total_candles, retries, executor)
        )
        if ohlcv_df.empty:
            logging.warning(f"Fetched OHLCV data is empty for {symbol} with timeframe {timeframe}")
        else:
            ohlcv_df.index = pd.to_datetime(ohlcv_df.index).tz_convert('UTC')
            ohlcv_df.reset_index(inplace=True)
            symbol_dir = create_symbol_dir(data_dir, symbol)
            candles_file = os.path.join(symbol_dir, f"{symbol}_candles.csv")
            try:
                ohlcv_df.to_csv(candles_file, index=False)
            except Exception as e:
                logging.error(f"Ошибка при сохранении данных в файл для {symbol}: {e}")
                raise APIRequestError(f"Ошибка при сохранении данных в файл: {e}")
        return ohlcv_df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol} with timeframe {timeframe}: {e}")
        raise APIRequestError(f"Error fetching data: {e}")
    finally:
        executor.shutdown(wait=False)


def fetch_data_with_metrics(api_key, api_secret, base_url, symbol, timeframe='5m', limit=1000, total_candles=2000, retries=5, data_dir="data"):
    base_url = "https://api.bybit.com"
    df = fetch_data(api_key, api_secret, base_url, symbol, timeframe, limit, total_candles, retries, data_dir)
    return df

def calculate_atr(df, period=14):
    """Calculate the Average True Range using high, low and close prices."""
    try:
        df = df.sort_index()
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        period = min(period, len(tr)) if len(tr) > 0 else 1
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr
    except Exception as e:
        logging.error(f"Error calculating ATR: {e}")
        return pd.Series()

@retry_on_failure()
def set_trading_stop(base_url, api_key, api_secret, symbol, stop_loss_price=0, take_profit_price=0, trailing_stop=0, tpsl_mode='Full',
                    tp_size=None, sl_size=None, tp_limit_price=None, sl_limit_price=None, default_config=None):
    endpoint = '/v5/position/trading-stop'
    params = {
        'category': 'linear',
        'symbol': symbol,
        'positionIdx': 0,
        'tpslMode': tpsl_mode,
    }

    if stop_loss_price > 0:
        params['stopLoss'] = str(stop_loss_price)
        params['slTriggerBy'] = 'LastPrice'
    else:
        params["stopLoss"] = '0'

    if take_profit_price > 0:
        params['takeProfit'] = str(take_profit_price)
        params['tpTriggerBy'] = 'LastPrice'
    else:
        params['takeProfit'] = '0'

    use_trailing_stop = default_config.get("USE_TRAILING_STOP", False) if default_config else False
    if use_trailing_stop and trailing_stop > 0:
        params['trailingStop'] = str(trailing_stop)
        params['activePrice'] = str(stop_loss_price)
    else:
        params['trailingStop'] = '0'
        params['activePrice'] = '0'

    if tpsl_mode == 'Partial':
        if tp_size is None or sl_size is None:
            logging.error("For Partial TP/SL mode, tp_size and sl_size must be provided.")
            raise APIRequestError("For Partial TP/SL mode, tp_size and sl_size must be provided.")

        if tp_size != sl_size:
            raise APIRequestError("tp_size and sl_size MUST be equal")

        params['tpSize'] = str(tp_size)
        params['slSize'] = str(sl_size)

        if tp_limit_price:
            params['tpLimitPrice'] = str(tp_limit_price)

        if sl_limit_price:
            params['slLimitPrice'] = str(sl_limit_price)

        params['tpOrderType'] = 'Limit' if tp_limit_price else 'Market'
        params['slOrderType'] = 'Limit' if sl_limit_price else 'Market'

    else:
        params['tpOrderType'] = 'Market'
        params['slOrderType'] = 'Market'

    params['api_key'] = api_key
    params['timestamp'] = int(time.time() * 1000)
    params['sign'] = generate_signature(params, api_secret)

    try:
        response = requests.post(base_url + endpoint, json=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result.get('retCode') != 0:
            ret_msg = result.get('retMsg', '').lower()
            if "trading stop already exists" in ret_msg:
                logging.warning(f"set_trading_stop: {result.get('retMsg')}")
                return result
            else:
                raise APIRequestError(f"API Error: {result.get('retMsg')}")
        logging.info(f"Торговый стоп установлен для {symbol}: {result}")
        return result

    except requests.exceptions.Timeout:
        logging.error(f"Timeout error in set_trading_stop for {symbol}")
        raise APIRequestError("Timeout during request")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error in set_trading_stop: {e}")
        raise APIRequestError(f"Request error: {e}")
    except Exception as e:
        logging.error(f"Error in set_trading_stop for {symbol}: {e}")
        raise APIRequestError(f"Error: {e}")

def get_min_order_size(base_url, symbol):
    @retry_on_failure()
    def inner():
        try:
            endpoint = '/v5/market/instruments-info'
            params = {'category': 'linear', 'symbol': symbol}
            response = requests.get(base_url + endpoint, params=params, timeout=10)
            response.raise_for_status()
            symbols_info = response.json()
            for info in symbols_info.get('result', {}).get('list', []):
                if info.get('symbol') == symbol:
                    min_order_qty = float(info['lotSizeFilter']['minOrderQty'])
                    logging.info(f"Min order quantity for {symbol}: {min_order_qty}")
                    return min_order_qty
            raise APIRequestError(f"Symbol {symbol} not found")
        except requests.exceptions.Timeout:
            logging.error(f"Timeout error in get_min_order_size for {symbol}")
            raise APIRequestError("Timeout during request")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error in get_min_order_size: {e}")
            raise APIRequestError(f"Request error: {e}")
        except (ValueError, KeyError, TypeError) as ve:
            logging.error(f"Value error in get_min_order_size: {ve}")
            raise APIRequestError(f"Value error: {ve}")
    return inner()

def get_max_qty(base_url, symbol):
    @retry_on_failure()
    def inner():
        try:
            endpoint = '/v5/market/instruments-info'
            params = {'category': 'linear', 'symbol': symbol}
            response = requests.get(base_url + endpoint, params=params, timeout=10)
            response.raise_for_status()
            instruments_info = response.json()

            for info in instruments_info.get('result', {}).get('list', []):
                if info.get('symbol') == symbol:
                    max_qty = float(info['lotSizeFilter']['maxOrderQty'])
                    return max_qty

            logging.error(f"Symbol {symbol} not found in instrument info")
            raise APIRequestError(f"Symbol {symbol} not found")
        except requests.exceptions.Timeout:
            logging.error(f"Timeout error in get_max_qty for {symbol}")
            raise APIRequestError("Timeout during request")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error in get_max_qty: {e}")
            raise APIRequestError(f"Request error: {e}")
        except (ValueError, KeyError, TypeError) as ve:
            logging.error(f"Value error in get_max_qty: {ve}")
            raise APIRequestError(f"Value error: {ve}")
    return inner()

@retry_on_failure()
def get_account_balance(base_url, api_key, api_secret):
    try:
        endpoint = '/v5/account/wallet-balance'
        params = {
            'accountType': 'UNIFIED',
            'api_key': api_key,
            'timestamp': int(time.time() * 1000),
            'recv_window': 5000
        }
        params['sign'] = generate_signature(params, api_secret)
        response = requests.get(base_url + endpoint, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result.get('retCode') != 0:
            raise APIRequestError(f"API Error: {result.get('retMsg')}")

        for item in result.get('result', {}).get('list', []):
            for coin in item.get('coin', []):
                if coin.get('coin') == 'USDT':
                    available_str = coin.get('availableToWithdraw', '')
                    if available_str:
                        available = float(available_str)
                    else:
                        available_str = coin.get('walletBalance', '0')
                        if available_str:
                            available = float(available_str)
                        else:
                            available = 0.0
                    return available

        raise APIRequestError("USDT balance not found in response")

    except requests.exceptions.Timeout:
        logging.error(f"Timeout error in get_account_balance")
        raise APIRequestError("Timeout during request")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error in get_account_balance: {e}")
        raise APIRequestError(f"Request error: {e}")
    except (ValueError, KeyError, TypeError) as e:
        logging.error(f"Error in get_account_balance: {e}")
        raise APIRequestError(f"Error: {e}")


@retry_on_failure()
def open_position_api(base_url, api_key, api_secret, symbol, side, qty, default_config=None):
    min_trade_qty = get_min_order_size(base_url, symbol)
    max_trade_qty = get_max_qty(base_url, symbol)
    try:
        qty = max(min_trade_qty, math.floor(float(qty) / min_trade_qty) * min_trade_qty)
        qty = round(qty, 6)
        if qty > max_trade_qty:
            logging.error(f"Order quantity {qty} exceeds maximum allowed {max_trade_qty} for {symbol}. Correcting to max allowed.")
            qty = max_trade_qty

        endpoint = '/v5/order/create'
        params = {
            'category': 'linear',
            'symbol': symbol,
            'side': side,
            'orderType': 'Market',
            'qty': str(qty),
            'timeInForce': 'IOC',
            'api_key': api_key,
            'timestamp': int(time.time() * 1000)
        }
        params['sign'] = generate_signature(params, api_secret)
        response = requests.post(base_url + endpoint, json=params, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result.get('retCode') != 0:
            raise APIRequestError(f"API error: {result.get('retMsg')}")

        logging.info(f"Position opened for {symbol}: {result}")
        return result

    except Exception as e:
        logging.error(f"Error during open_position_api for {symbol}: {e}")
        raise APIRequestError(f"Error during open_position_api: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    api_key = "1"
    api_secret = "2"
    base_url = "https://api.bybit.com"
    symbol = "BTCUSDT"
    timeframe = "5m"
    
    df = fetch_data(api_key, api_secret, base_url, symbol, timeframe, limit=1000, total_candles=13500, retries=5, data_dir="data")
    print(len(df))
    print(df.head())