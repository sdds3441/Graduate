import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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
data = pd.read_csv('./data/TSdata/Processed/whole_data.csv', header=None)

# 데이터에서 첫 번째 열(Date)과 마지막 열(Average_Price)만 제외하고 나머지 숫자 열 선택
X = data.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').values
Y = pd.to_numeric(data[6], errors='coerce').values

# NaN 값 제거
nan_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
X = X[nan_mask]
Y = Y[nan_mask]

# 데이터 정규화 (LSTM에서 일반적으로 사용)
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))

# 타임 스텝 길이 설정 (예: 10개 이전 데이터 사용)
time_steps = 10

# 데이터셋을 타임 스텝으로 변환하는 함수
def create_sequences(X, Y, time_steps=1):
    x_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        x_seq.append(X[i:i + time_steps])
        y_seq.append(Y[i + time_steps])
    return np.array(x_seq), np.array(y_seq)

# 타임 스텝 적용
x_seq, y_seq = create_sequences(X_scaled, Y_scaled, time_steps)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size=0.2, random_state=7)

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
model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_test, y_test), verbose=2, shuffle=False, callbacks=[early_stopping])

# 예측
y_pred = model.predict(x_test)

# 정규화된 값 역변환
y_test_inverse = scaler_Y.inverse_transform(y_test)
y_pred_inverse = scaler_Y.inverse_transform(y_pred)

# 평가 (예: 평균 절대 오차)
mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
print(f'Mean Absolute Error: {mae}')

y_pred_inverse = scaler_Y.inverse_transform(y_pred)

# 결과 출력
print(y_pred_inverse)
