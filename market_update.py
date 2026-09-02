import os
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
import mplfinance as mpf

# ====== KONFIGURASI ======
SYMBOLS_TO_TRY = ["XAUUSD=X", "GC=F"]  # fallback kalau simbol pertama kosong/gagal
INTERVAL = "5m"
LOOKBACK = "5d"
RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21
CANDLES_FOR_CHART = 60  # jumlah candle terakhir yang ditampilkan di chart (60 x 5m = 5 jam)
CHART_PATH = "market_chart.png"

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]


def get_price_data():
    last_error = None
    for symbol in SYMBOLS_TO_TRY:
        try:
            df = yf.download(
                symbol, period=LOOKBACK, interval=INTERVAL, progress=False, auto_adjust=False
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) > EMA_SLOW + RSI_PERIOD:
                return df, symbol
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Gagal mengambil data harga. Error terakhir: {last_error}")


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def detect_candle_patterns(df: pd.DataFrame):
    """Deteksi pola candlestick sederhana dari candle terakhir (dibanding candle
    sebelumnya). Ini deteksi berbasis aturan standar candlestick, BUKAN sinyal
    trading — sekadar deskripsi bentuk candle yang terbentuk."""
    patterns = []
    if len(df) < 2:
        return patterns

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    def body(row):
        return abs(row["Close"] - row["Open"])

    def range_(row):
        return row["High"] - row["Low"]

    def upper_wick(row):
        return row["High"] - max(row["Close"], row["Open"])

    def lower_wick(row):
        return min(row["Close"], row["Open"]) - row["Low"]

    curr_body = body(curr)
    curr_range = range_(curr) if range_(curr) > 0 else 1e-9
    curr_bullish = curr["Close"] > curr["Open"]
    curr_bearish = curr["Close"] < curr["Open"]
    prev_bullish = prev["Close"] > prev["Open"]
    prev_bearish = prev["Close"] < prev["Open"]

    if curr_bullish and prev_bearish and curr["Close"] >= prev["Open"] and curr["Open"] <= prev["Close"]:
        patterns.append("Bullish Engulfing")
    if curr_bearish and prev_bullish and curr["Open"] >= prev["Close"] and curr["Close"] <= prev["Open"]:
        patterns.append("Bearish Engulfing")
    if curr_body / curr_range < 0.1:
        patterns.append("Doji")
    if lower_wick(curr) > 2 * curr_body and upper_wick(curr) < curr_body:
        patterns.append("Hammer")
    if upper_wick(curr) > 2 * curr_body and lower_wick(curr) < curr_body:
        patterns.append("Shooting Star")

    return patterns


def get_trend_bias(df: pd.DataFrame, latest_rsi: float) -> str:
    """Ringkasan arah berdasarkan posisi EMA9/EMA21 & RSI.
    Ini murni rangkuman indikator teknikal saat ini, BUKAN prediksi atau
    jaminan arah harga ke depan."""
    ema_fast = df["Close"].ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
    ema_slow = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]
    price = df["Close"].iloc[-1]

    if ema_fast > ema_slow and price > ema_fast and latest_rsi > 50:
        return "Bullish (harga & EMA9 di atas EMA21, RSI > 50)"
    elif ema_fast < ema_slow and price < ema_fast and latest_rsi < 50:
        return "Bearish (harga & EMA9 di bawah EMA21, RSI < 50)"
    else:
        return "Netral / Sideways (indikator belum searah)"


def make_chart(df: pd.DataFrame, symbol: str) -> None:
    plot_df = df.tail(CANDLES_FOR_CHART).copy()
    rsi = calculate_rsi(plot_df["Close"], RSI_PERIOD)
    ema_fast = plot_df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = plot_df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

    add_plots = [
        mpf.make_addplot(ema_fast, color="dodgerblue", width=1.0),
        mpf.make_addplot(ema_slow, color="orange", width=1.0),
        mpf.make_addplot(rsi, panel=1, color="purple", ylabel="RSI", secondary_y=False),
        mpf.make_addplot([70] * len(plot_df), panel=1, color="red", linestyle="--", width=0.7, secondary_y=False),
        mpf.make_addplot([20] * len(plot_df), panel=1, color="green", linestyle="--", width=0.7, secondary_y=False),
    ]

    mpf.plot(
        plot_df,
        type="candle",
        style="yahoo",
        addplot=add_plots,
        panel_ratios=(3, 1),
        title=f"\n{symbol} - M5",
        ylabel="Harga",
        volume=False,
        savefig=dict(fname=CHART_PATH, dpi=150, bbox_inches="tight"),
    )


def send_telegram_photo(caption: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(CHART_PATH, "rb") as photo:
        files = {"photo": photo}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
        response = requests.post(url, data=data, files=files, timeout=30)
        response.raise_for_status()


def main():
    df, symbol_used = get_price_data()
    close = df["Close"]
    rsi_series = calculate_rsi(close, RSI_PERIOD)
    latest_rsi = float(rsi_series.iloc[-1])
    latest_price = float(close.iloc[-1])
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    bias = get_trend_bias(df, latest_rsi)
    patterns = detect_candle_patterns(df)
    pattern_text = ", ".join(patterns) if patterns else "Tidak ada pola signifikan terdeteksi"

    make_chart(df, symbol_used)

    caption = (
        "📊 <b>Market Update - XAUUSD</b>\n"
        f"Simbol: {symbol_used}\n"
        f"Harga: {latest_price:.2f}\n"
        f"RSI (14, M5): {latest_rsi:.2f}\n"
        f"Bias Teknikal: {bias}\n"
        f"Pola Candle Terbaru: {pattern_text}\n"
        f"Waktu: {now_str}\n\n"
        "⚠️ Ringkasan indikator teknikal otomatis, bukan rekomendasi atau jaminan arah harga."
    )

    print(caption.replace("\n", " | "))
    send_telegram_photo(caption)


if __name__ == "__main__":
    main()
