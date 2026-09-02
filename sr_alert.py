import os
import json
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

# ====== KONFIGURASI ======
SYMBOLS_TO_TRY = ["XAUUSD=X", "GC=F"]
INTERVAL = "15m"
LOOKBACK = "60d"          # maksimal history intraday 15m yang diizinkan Yahoo Finance
SWING_WINDOW = 3          # candle kiri/kanan yang harus lebih rendah/tinggi supaya jadi swing point
CLUSTER_TOLERANCE_PCT = 0.0015   # 0.15% -> swing point yang berdekatan digabung jadi satu level
TOUCH_TOLERANCE_PCT = 0.0015     # 0.15% -> jarak dianggap "menyentuh" level
RESET_PCT = 0.005                # 0.5% -> harga harus menjauh sejauh ini dulu sebelum level yg sama boleh alert lagi
MAX_LEVELS_EACH_SIDE = 3         # jumlah level yang ditampilkan di pesan (bukan cuma yang terdekat)
STATE_FILE = "sr_state.json"

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
            if not df.empty and len(df) > SWING_WINDOW * 2 + 5:
                return df, symbol
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Gagal mengambil data harga. Error terakhir: {last_error}")


def find_swing_points(df: pd.DataFrame, window: int = 3):
    """Cari swing high/low ala indikator fractal: sebuah candle dianggap
    swing point kalau High/Low-nya paling ekstrem dibanding `window` candle
    di kiri DAN kanannya."""
    highs = df["High"]
    lows = df["Low"]
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
    """Gabungkan harga-harga yang berdekatan (dalam tol_pct) jadi satu level.
    Mengembalikan list (harga_rata_rata, jumlah_sentuhan)."""
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


def get_sr_levels(df: pd.DataFrame, current_price: float):
    swing_highs, swing_lows = find_swing_points(df, SWING_WINDOW)
    resistance_clusters = cluster_levels(swing_highs, CLUSTER_TOLERANCE_PCT)
    support_clusters = cluster_levels(swing_lows, CLUSTER_TOLERANCE_PCT)

    resistances = sorted(
        [c for c in resistance_clusters if c[0] > current_price], key=lambda c: c[0]
    )[:MAX_LEVELS_EACH_SIDE]
    supports = sorted(
        [c for c in support_clusters if c[0] < current_price], key=lambda c: -c[0]
    )[:MAX_LEVELS_EACH_SIDE]
    return supports, resistances


def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
            return data.get("active_type"), data.get("active_level")
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None


def save_state(active_type, active_level):
    with open(STATE_FILE, "w") as f:
        json.dump({"active_type": active_type, "active_level": active_level}, f, indent=2)


def format_levels(levels, label):
    if not levels:
        return f"{label}: tidak ada level terdeteksi"
    lines = [f"{price:.2f} (disentuh {touches}x)" for price, touches in levels]
    return f"{label}:\n  " + "\n  ".join(lines)


def send_telegram_message(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    response = requests.post(url, data=payload, timeout=15)
    response.raise_for_status()


def main():
    df, symbol_used = get_price_data()
    current_price = float(df["Close"].iloc[-1])
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    supports, resistances = get_sr_levels(df, current_price)
    nearest_support = supports[0] if supports else None
    nearest_resistance = resistances[0] if resistances else None

    touched = None
    if nearest_support and abs(current_price - nearest_support[0]) / current_price <= TOUCH_TOLERANCE_PCT:
        touched = ("support", nearest_support[0], nearest_support[1])
    elif nearest_resistance and abs(current_price - nearest_resistance[0]) / current_price <= TOUCH_TOLERANCE_PCT:
        touched = ("resistance", nearest_resistance[0], nearest_resistance[1])

    active_type, active_level = load_state()

    print(
        f"[{now_str}] Symbol={symbol_used} Price={current_price:.2f} "
        f"NearestSupport={nearest_support} NearestResistance={nearest_resistance} "
        f"Touched={touched} ActiveState=({active_type}, {active_level})"
    )

    if touched:
        level_type, level_price, touches = touched
        same_as_active = (
            active_type == level_type
            and active_level is not None
            and abs(level_price - active_level) / level_price <= CLUSTER_TOLERANCE_PCT
        )
        if not same_as_active:
            emoji = "🟩" if level_type == "support" else "🟥"
            label = "SUPPORT" if level_type == "support" else "RESISTANCE"
            text = (
                f"{emoji} <b>HARGA MENDEKATI {label}</b>\n"
                f"Simbol: {symbol_used} (M15)\n"
                f"Harga saat ini: {current_price:.2f}\n"
                f"Level {label.lower()}: {level_price:.2f} (tersentuh {touches}x sebelumnya)\n"
                f"Waktu: {now_str}\n\n"
                f"{format_levels(supports, 'Support terdekat')}\n"
                f"{format_levels(resistances, 'Resistance terdekat')}\n\n"
                "⚠️ Level dihitung otomatis dari swing high/low 60 hari terakhir (M15). "
                "Bukan jaminan harga akan pantul/tembus di titik ini."
            )
            try:
                send_telegram_message(text)
                print("Notifikasi S/R terkirim.")
            except Exception as e:
                print(f"Gagal mengirim notifikasi Telegram: {e}")
            save_state(level_type, level_price)
        else:
            print("Masih di level aktif yang sama, tidak kirim ulang.")
    else:
        # kalau sudah cukup jauh dari level yang terakhir aktif, reset supaya
        # nanti kalau balik lagi ke level itu bisa alert lagi
        if active_level is not None:
            dist = abs(current_price - active_level) / current_price
            if dist >= RESET_PCT:
                save_state(None, None)
                print("Sudah menjauh dari level aktif, state direset.")


if __name__ == "__main__":
    main()
