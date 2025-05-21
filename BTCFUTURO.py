import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

# Descargar datos desde 2023 hasta hoy
df = yf.download('BTC-USD', start='2023-01-01')

# Crear características desfasadas
df['Close_t-1'] = df['Close'].shift(1)
df['Close_t-2'] = df['Close'].shift(2)
df = df.dropna()

# Entradas (features) y salida (target)
X = df[['Close_t-1', 'Close_t-2']]
y = df['Close']

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar el modelo en todo el histórico disponible
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_scaled, y)

# Últimos dos valores para empezar la predicción
last_close_1 = df['Close'].iloc[-1]
last_close_2 = df['Close'].iloc[-2]

# Generar predicción para 30 días
future_preds = []

for _ in range(30):
    last_close_1 = float(last_close_1)
    last_close_2 = float(last_close_2)

    input_scaled = scaler.transform([[last_close_1, last_close_2]])
    next_price = model.predict(input_scaled)[0]
    future_preds.append(next_price)

    last_close_2 = last_close_1
    last_close_1 = next_price
    
    
# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Predicción
y_pred = model.predict(X_test)
    
# Evaluación
y_test_np = y_test.values.flatten()
mse = np.mean((y_test_np - y_pred)**2)
mae = np.mean(np.abs(y_test_np - y_pred))
error_pct = np.mean(np.abs((y_test_np - y_pred) / y_test_np)) * 100

# Crear índice de fechas futuras
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

# Crear DataFrame con resultados
future_df = pd.DataFrame({'Fecha': future_dates, 'Predicción BTC': future_preds})
future_df.set_index('Fecha', inplace=True)

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"Error porcentual medio: {error_pct:.2f}%")

# Mostrar gráfico
plt.figure(figsize=(12, 5))
plt.plot(df['Close'].iloc[-60:], label='Precio histórico')
plt.plot(future_df['Predicción BTC'], label='Predicción 30 días', linestyle='dashed')
plt.title('Predicción del precio del BTC para los próximos 30 días')
plt.xlabel('Fecha')
plt.ylabel('Precio (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
