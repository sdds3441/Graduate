import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator

# GPU 메모리 설정
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# 데이터 로드
data = pd.read_csv('./data/TSdata/Processed/whole_data.csv')

# 데이터에서 첫 번째 열(Date)을 제외하고 나머지 열 선택
X = data[['Avg_temp', 'Temp_diff']].apply(pd.to_numeric, errors='coerce').values
Y = pd.to_numeric(data['Consumer_price'], errors='coerce').values
dates = pd.to_datetime(data['Date'], format='%Y-%m-%d')

# NaN 값 제거
nan_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
X = X[nan_mask]
Y = Y[nan_mask]
dates = dates[nan_mask]  # NaN 제거된 날짜 데이터도 동일하게 적용

# 데이터 정규화 (LSTM에서 일반적으로 사용)
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))

# 타임 스텝 길이 설정 (예: 10개 이전 데이터 사용)
time_steps = 10

# 데이터셋을 타임 스텝으로 변환하는 함수 (데이터 일관성 유지)
def create_sequences(X, Y, dates, time_steps=1):
    x_seq, y_seq, date_seq = [], [], []
    for i in range(len(X) - time_steps):
        x_seq.append(X[i:i + time_steps])
        y_seq.append(Y[i + time_steps])
        date_seq.append(dates.iloc[i + time_steps])  # 날짜 시퀀스 정확히 유지
    return np.array(x_seq), np.array(y_seq), np.array(date_seq)

# 타임 스텝 적용
x_seq, y_seq, date_seq = create_sequences(X_scaled, Y_scaled, dates, time_steps)

# 데이터를 시간 순서대로 나누기 (80%는 과거 학습, 20%는 미래 테스트)
split_index = int(len(x_seq) * 0.8)
x_train, x_test = x_seq[:split_index], x_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]
date_train, date_test = date_seq[:split_index], date_seq[split_index:]

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dense(1))  # 출력층 (실수값 하나)

# 모델 컴파일 (회귀 문제에 맞게 'mse' 사용), 학습률 조정
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

# 조기 종료 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 모델 학습
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), verbose=2, shuffle=False, callbacks=[early_stopping])

# 예측
y_pred = model.predict(x_test)

# 정규화된 값 역변환
y_test_inverse = scaler_Y.inverse_transform(y_test)
y_pred_inverse = scaler_Y.inverse_transform(y_pred)

# 평가 (예: 평균 절대 오차)
mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
print(f'Mean Absolute Error: {mae}')

# 예측값과 실제값 시각화
plt.figure(figsize=(10, 6))
plt.plot(date_test, y_test_inverse, label='Actual Prices', color='blue', linewidth=2)
plt.plot(date_test, y_pred_inverse, label='Predicted Prices', color='orange', linestyle='-', linewidth=2)
plt.title('Actual vs Predicted Consumer Prices')
plt.xlabel('Date')
plt.ylabel('Consumer Price')
plt.legend()

# 연도별 첫 번째 날짜만 x축에 표시하고, '20'을 제외한 2자리 연도 표시
years = YearLocator()  # 연도별 주요 표시
years_fmt = DateFormatter('%y')  # 두 자리 연도만 표시 ('14', '15', '16' 등)

plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(years_fmt)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 결과 출력
print(y_pred_inverse)
