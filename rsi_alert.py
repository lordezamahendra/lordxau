import os
import json
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

# ====== KONFIGURASI ======
RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 70
SYMBOLS_TO_TRY = ["XAUUSD=X", "GC=F"]  # fallback kalau simbol pertama kosong/gagal
INTERVAL = "5m"
LOOKBACK = "5d"
STATE_FILE = "state.json"

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]


def get_price_data():
    """Ambil data candle terbaru dari Yahoo Finance. Coba beberapa simbol
    sebagai fallback kalau salah satu tidak tersedia."""
    last_error = None
    for symbol in SYMBOLS_TO_TRY:
        try:
            df = yf.download(
                symbol,
                period=LOOKBACK,
                interval=INTERVAL,
                progress=False,
                auto_adjust=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) > RSI_PERIOD + 1:
                return df, symbol
        except Exception as e:  # coba simbol berikutnya
            last_error = e
    raise RuntimeError(
        f"Gagal mengambil data harga untuk semua simbol. Error terakhir: {last_error}"
    )


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Hitung RSI dengan metode Wilder's smoothing (standar)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_zone(rsi_value: float) -> str:
    if rsi_value <= RSI_OVERSOLD:
        return "oversold"
    if rsi_value >= RSI_OVERBOUGHT:
        return "overbought"
    return "normal"


def load_last_zone() -> str:
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f).get("last_zone", "normal")
    except (FileNotFoundError, json.JSONDecodeError):
        return "normal"


def save_last_zone(zone: str, rsi_value: float, price: float, timestamp: str) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(
            {
                "last_zone": zone,
                "rsi": round(rsi_value, 2),
                "price": round(price, 2),
                "updated_at": timestamp,
            },
            f,
            indent=2,
        )


def send_telegram_message(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    response = requests.post(url, data=payload, timeout=15)
    response.raise_for_status()


def main():
    df, symbol_used = get_price_data()
    close = df["Close"]
    rsi_series = calculate_rsi(close, RSI_PERIOD)

    latest_rsi = float(rsi_series.iloc[-1])
    latest_price = float(close.iloc[-1])
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    current_zone = get_zone(latest_rsi)
    last_zone = load_last_zone()

    print(
        f"[{now_str}] Symbol={symbol_used} Price={latest_price:.2f} "
        f"RSI={latest_rsi:.2f} Zone={current_zone} (sebelumnya: {last_zone})"
    )

    # Hanya kirim notifikasi saat BARU masuk ke zona oversold/overbought,
    # supaya tidak spam tiap 5 menit selama RSI masih di zona yang sama.
    if current_zone != last_zone and current_zone in ("oversold", "overbought"):
        if current_zone == "oversold":
            text = (
                "🟢 <b>RSI OVERSOLD</b>\n"
                f"Simbol: {symbol_used} (Gold/XAUUSD)\n"
                f"Harga: {latest_price:.2f}\n"
                f"RSI (14, M5): {latest_rsi:.2f}\n"
                f"Waktu: {now_str}\n\n"
                "RSI turun ke area ≤ 20 (potensi jenuh jual)."
            )
        else:
            text = (
                "🔴 <b>RSI OVERBOUGHT</b>\n"
                f"Simbol: {symbol_used} (Gold/XAUUSD)\n"
                f"Harga: {latest_price:.2f}\n"
                f"RSI (14, M5): {latest_rsi:.2f}\n"
                f"Waktu: {now_str}\n\n"
                "RSI naik ke area ≥ 70 (potensi jenuh beli)."
            )
        try:
            send_telegram_message(text)
            print("Notifikasi Telegram terkirim.")
        except Exception as e:
            print(f"Gagal mengirim notifikasi Telegram: {e}")

    save_last_zone(current_zone, latest_rsi, latest_price, now_str)


if __name__ == "__main__":
    main()
