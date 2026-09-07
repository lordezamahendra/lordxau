import os
import json
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
import mplfinance as mpf

# ====== KONFIGURASI ======
SYMBOLS_TO_TRY = ["XAUUSD=X", "GC=F"]

M5_INTERVAL, M5_LOOKBACK = "5m", "5d"
M15_INTERVAL, M15_LOOKBACK = "15m", "60d"

RSI_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21

SWING_WINDOW = 3
CLUSTER_TOLERANCE_PCT = 0.0015
TOUCH_TOLERANCE_PCT = 0.0025

CANDLES_FOR_CHART = 60
CHART_PATH = "signal_chart.png"
STATE_FILE = "signal_state.json"

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]


# ---------- Ambil data ----------
def get_price_data(interval, lookback, min_len):
    last_error = None
    for symbol in SYMBOLS_TO_TRY:
        try:
            df = yf.download(symbol, period=lookback, interval=interval, progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) > min_len:
                return df, symbol
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Gagal mengambil data ({interval}). Error terakhir: {last_error}")


# ---------- Indikator ----------
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def find_swing_points(df: pd.DataFrame, window: int = 3):
    highs, lows = df["High"], df["Low"]
    swing_highs, swing_lows = [], []
    n = len(df)
    for i in range(window, n - window):
        wh = highs.iloc[i - window : i + window + 1]
        wl = lows.iloc[i - window : i + window + 1]
        if highs.iloc[i] == wh.max():
            swing_highs.append(float(highs.iloc[i]))
        if lows.iloc[i] == wl.min():
            swing_lows.append(float(lows.iloc[i]))
    return swing_highs, swing_lows


def cluster_levels(prices, tol_pct: float):
    if not prices:
        return []
    prices = sorted(prices)
    clusters = [[prices[0]]]
    for p in prices[1:]:
        if abs(p - clusters[-1][-1]) / clusters[-1][-1] <= tol_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [(sum(c) / len(c), len(c)) for c in clusters]


def get_sr_levels(df_m15: pd.DataFrame, current_price: float):
    swing_highs, swing_lows = find_swing_points(df_m15, SWING_WINDOW)
    resistance_clusters = cluster_levels(swing_highs, CLUSTER_TOLERANCE_PCT)
    support_clusters = cluster_levels(swing_lows, CLUSTER_TOLERANCE_PCT)
    resistances = sorted([c for c in resistance_clusters if c[0] > current_price], key=lambda c: c[0])[:3]
    supports = sorted([c for c in support_clusters if c[0] < current_price], key=lambda c: -c[0])[:3]
    return supports, resistances


def detect_candle_patterns(df: pd.DataFrame):
    if len(df) < 2:
        return []
    prev, curr = df.iloc[-2], df.iloc[-1]
    patterns = []

    def body(r): return abs(r["Close"] - r["Open"])
    def rng(r): return r["High"] - r["Low"]
    def upper_wick(r): return r["High"] - max(r["Close"], r["Open"])
    def lower_wick(r): return min(r["Close"], r["Open"]) - r["Low"]

    cb = body(curr)
    cr = rng(curr) if rng(curr) > 0 else 1e-9
    cbull, cbear = curr["Close"] > curr["Open"], curr["Close"] < curr["Open"]
    pbull, pbear = prev["Close"] > prev["Open"], prev["Close"] < prev["Open"]

    if cbull and pbear and curr["Close"] >= prev["Open"] and curr["Open"] <= prev["Close"]:
        patterns.append("Bullish Engulfing")
    if cbear and pbull and curr["Open"] >= prev["Close"] and curr["Close"] <= prev["Open"]:
        patterns.append("Bearish Engulfing")
    if cb / cr < 0.1:
        patterns.append("Doji")
    if lower_wick(curr) > 2 * cb and upper_wick(curr) < cb:
        patterns.append("Hammer")
    if upper_wick(curr) > 2 * cb and lower_wick(curr) < cb:
        patterns.append("Shooting Star")
    return patterns


# ---------- Confluence signal ----------
def compute_signal(df_m5: pd.DataFrame, supports, resistances):
    close = df_m5["Close"]
    price = float(close.iloc[-1])
    rsi = float(calculate_rsi(close, RSI_PERIOD).iloc[-1])
    ema_fast = float(close.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1])
    ema_slow = float(close.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1])
    patterns = detect_candle_patterns(df_m5)

    bull, bear = [], []

    if ema_fast > ema_slow:
        bull.append("EMA9 > EMA21 (tren M5 naik)")
    elif ema_fast < ema_slow:
        bear.append("EMA9 < EMA21 (tren M5 turun)")

    if price > ema_fast:
        bull.append("Harga di atas EMA9")
    elif price < ema_fast:
        bear.append("Harga di bawah EMA9")

    if rsi > 55:
        bull.append(f"RSI {rsi:.1f} > 55 (momentum naik)")
    elif rsi < 45:
        bear.append(f"RSI {rsi:.1f} < 45 (momentum turun)")

    nearest_support = supports[0] if supports else None
    nearest_resistance = resistances[0] if resistances else None
    if nearest_support and abs(price - nearest_support[0]) / price <= TOUCH_TOLERANCE_PCT:
        bull.append(f"Harga dekat support {nearest_support[0]:.2f}")
    if nearest_resistance and abs(price - nearest_resistance[0]) / price <= TOUCH_TOLERANCE_PCT:
        bear.append(f"Harga dekat resistance {nearest_resistance[0]:.2f}")

    bullish_patterns = [p for p in patterns if p in ("Bullish Engulfing", "Hammer")]
    bearish_patterns = [p for p in patterns if p in ("Bearish Engulfing", "Shooting Star")]
    if bullish_patterns:
        bull.append(f"Pola candle: {', '.join(bullish_patterns)}")
    if bearish_patterns:
        bear.append(f"Pola candle: {', '.join(bearish_patterns)}")

    bull_score, bear_score = len(bull), len(bear)

    if bull_score >= 4:
        label = "BUY - Confluence Kuat"
    elif bull_score == 3 and bull_score > bear_score:
        label = "BUY - Confluence Sedang"
    elif bear_score >= 4:
        label = "SELL - Confluence Kuat"
    elif bear_score == 3 and bear_score > bull_score:
        label = "SELL - Confluence Sedang"
    else:
        label = "NETRAL - Belum Ada Confluence Jelas"

    return {
        "label": label,
        "price": price,
        "rsi": rsi,
        "bull": bull,
        "bear": bear,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "patterns": patterns,
    }


# ---------- Chart ----------
def make_chart(df_m5: pd.DataFrame, symbol: str, nearest_support, nearest_resistance):
    plot_df = df_m5.tail(CANDLES_FOR_CHART).copy()
    rsi = calculate_rsi(plot_df["Close"], RSI_PERIOD)
    ema_fast = plot_df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = plot_df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

    add_plots = [
        mpf.make_addplot(ema_fast, color="dodgerblue", width=1.0),
        mpf.make_addplot(ema_slow, color="orange", width=1.0),
        mpf.make_addplot(rsi, panel=1, color="purple", ylabel="RSI", secondary_y=False),
        mpf.make_addplot([70] * len(plot_df), panel=1, color="red", linestyle="--", width=0.6, secondary_y=False),
        mpf.make_addplot([20] * len(plot_df), panel=1, color="green", linestyle="--", width=0.6, secondary_y=False),
    ]

    hlines_values, hlines_colors = [], []
    if nearest_support:
        hlines_values.append(nearest_support[0])
        hlines_colors.append("green")
    if nearest_resistance:
        hlines_values.append(nearest_resistance[0])
        hlines_colors.append("red")

    mpf.plot(
        plot_df,
        type="candle",
        style="yahoo",
        addplot=add_plots,
        hlines=dict(hlines=hlines_values, colors=hlines_colors, linestyle="-.", linewidths=1.0) if hlines_values else None,
        panel_ratios=(3, 1),
        title=f"\n{symbol} - M5",
        ylabel="Harga",
        volume=False,
        savefig=dict(fname=CHART_PATH, dpi=150, bbox_inches="tight"),
    )


# ---------- State (anti-spam) ----------
def load_last_label():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f).get("last_label")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_last_label(label):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_label": label}, f, indent=2)


# ---------- Telegram ----------
def send_telegram_photo(caption: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(CHART_PATH, "rb") as photo:
        files = {"photo": photo}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
        response = requests.post(url, data=data, files=files, timeout=30)
        response.raise_for_status()


def main():
    df_m5, symbol_m5 = get_price_data(M5_INTERVAL, M5_LOOKBACK, EMA_SLOW + RSI_PERIOD)
    df_m15, _ = get_price_data(M15_INTERVAL, M15_LOOKBACK, SWING_WINDOW * 2 + 5)

    current_price = float(df_m5["Close"].iloc[-1])
    supports, resistances = get_sr_levels(df_m15, current_price)
    result = compute_signal(df_m5, supports, resistances)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    last_label = load_last_label()

    print(
        f"[{now_str}] Price={result['price']:.2f} RSI={result['rsi']:.1f} "
        f"Label={result['label']} (sebelumnya: {last_label})"
    )

    if result["label"] != last_label:
        support_txt = f"{result['nearest_support'][0]:.2f} (disentuh {result['nearest_support'][1]}x)" if result["nearest_support"] else "tidak ada"
        resistance_txt = f"{result['nearest_resistance'][0]:.2f} (disentuh {result['nearest_resistance'][1]}x)" if result["nearest_resistance"] else "tidak ada"
        pattern_txt = ", ".join(result["patterns"]) if result["patterns"] else "tidak ada pola signifikan"

        factor_lines = "\n".join([f"  ✅ {f}" for f in result["bull"]]) or "  (tidak ada)"
        bear_lines = "\n".join([f"  ❌ {f}" for f in result["bear"]]) or "  (tidak ada)"

        emoji = "🟢" if result["label"].startswith("BUY") else ("🔴" if result["label"].startswith("SELL") else "⚪")

        caption = (
            f"{emoji} <b>SINYAL: {result['label']}</b>\n"
            f"Simbol: {symbol_m5} (TF M5)\n"
            f"Harga: {result['price']:.2f}\n"
            f"RSI (14, M5): {result['rsi']:.1f}\n"
            f"Support terdekat: {support_txt}\n"
            f"Resistance terdekat: {resistance_txt}\n"
            f"Pola candle: {pattern_txt}\n"
            f"Waktu: {now_str}\n\n"
            f"<b>Faktor mendukung BUY:</b>\n{factor_lines}\n\n"
            f"<b>Faktor mendukung SELL:</b>\n{bear_lines}\n\n"
            "⚠️ Ini hasil skor rule-based dari indikator teknikal M5, BUKAN sinyal "
            "otomatis machine-learning dan BUKAN rekomendasi/jaminan profit. "
            "Tidak memperhitungkan berita/fundamental. Tetap gunakan manajemen risiko sendiri."
        )

        make_chart(df_m5, symbol_m5, result["nearest_support"], result["nearest_resistance"])
        try:
            send_telegram_photo(caption)
            print("Notifikasi sinyal terkirim.")
        except Exception as e:
            print(f"Gagal mengirim notifikasi Telegram: {e}")
    else:
        print("Label sinyal sama seperti sebelumnya, tidak kirim ulang.")

    save_last_label(result["label"])


if __name__ == "__main__":
    main()
