import pandas as pd
import numpy as np

# 1. Veriyi yükle ve zamana göre sırala
df = pd.read_csv('btc_1d_processed.csv')
df['Open time'] = pd.to_datetime(df['Open time'])
df = df.sort_values('Open time').reset_index(drop=True)

# 2. Getiriler (Returns)
df['Return'] = df['Close'].pct_change()
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# 3. Hareketli Ortalamalar
df['SMA_7'] = df['Close'].rolling(window=7).mean()
df['SMA_30'] = df['Close'].rolling(window=30).mean()
df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()

# 4. Volatilite (Son 7 günün getiri standart sapması)
df['Volatility_7d'] = df['Return'].rolling(window=7).std()

# 5. RSI (Relative Strength Index) Hesaplama
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI_14'] = calculate_rsi(df['Close'], window=14)

# 6. MACD Hesaplama
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']

# 7. Zamansal Özellikler
df['Day_of_Week'] = df['Open time'].dt.dayofweek # 0: Pazartesi, 6: Pazar
df['Month'] = df['Open time'].dt.month

# 8. Hareketli ortalamalar geçmiş verilere ihtiyaç duyduğundan ilk satırlarda NaN (boş) değerler oluşur.
# Bunları siliyoruz.
df_featured = df.dropna().copy()

# Yeni veriyi kaydet
df_featured.to_csv('btc_1d_featured.csv', index=False)