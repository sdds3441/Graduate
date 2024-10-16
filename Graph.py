import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
file_path = 'data/TSdata/Processed/Date_price.csv'  # 파일 경로 입력
data = pd.read_csv(file_path)

# 'Consumer_price' 열이 문자열일 경우에만 쉼표 제거
if data['Consumer_price'].dtype == 'object':
    data['Consumer_price'] = pd.to_numeric(data['Consumer_price'].str.replace(',', ''), errors='coerce')

# 결측값 제거
data_cleaned = data.dropna(subset=['Consumer_price'])

# 'Date' 열을 YYYYMMDD 형식에서 datetime 형식으로 변환
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%Y%m%d')

# 그래프 그리기
plt.figure(figsize=(10,6))
plt.plot(data_cleaned['Date'], data_cleaned['Consumer_price'], label='Consumer Price (Won)', color='red')
plt.title('Consumer Price Over Time')
plt.xlabel('Date')
plt.ylabel('Consumer Price')  # 세로 축에 단위 추가
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
