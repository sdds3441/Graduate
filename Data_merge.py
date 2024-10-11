# 필요한 라이브러리 불러오기
import pandas as pd

# 데이터 불러오기
temperature_data = pd.read_csv('./data/TSdata/Processed/merged_temperature_rainfall_data.csv')
rainfall_data = pd.read_csv('./data/TSdata/Processed/Monthly_Fluctuation_Range_and_Average_Price.csv')

# 'Date' 열을 기준으로 두 데이터셋 합치기
merged_data = pd.merge(temperature_data, rainfall_data, on='Date', how='inner')

merged_data.to_csv('./data/TSdata/Processed/whole_data.csv', index=False)
