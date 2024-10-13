import pandas as pd

# 1. CSV 파일을 읽어오면서 불필요한 메타데이터 행을 건너뜁니다.
file_path = 'data/TSdata/Monthly/ta_20241010140121.csv'
temperature_data = pd.read_csv(file_path, encoding='ISO-8859-1', skiprows=8, delimiter=',', header=None)

# 2. 첫 번째 열에 포함된 불필요한 탭 문자를 제거합니다.
temperature_data[0] = temperature_data[0].str.strip()
temperature_data[1] = temperature_data[1].str.strip()

print(temperature_data)
"""
# 3. 열 이름을 지정합니다.
temperature_data.columns = ['Year_Month', 'Location', 'Average_Temperature(°C)', 'Average_Min_Temperature(°C)', 'Average_Max_Temperature(°C)']

# 4. 영어로 변환된 열 이름을 매핑합니다.
temperature_data = temperature_data.rename(columns={
    'Year_Month': 'Date',
    'Average_Temperature(°C)': 'Avg',
    'Average_Min_Temperature(°C)': 'Avg_Min',
    'Average_Max_Temperature(°C)': 'Avg_Max'
})

# 5. 필요한 열인 'Date', 'Avg', 'Avg_Min', 'Avg_Max'만 선택합니다.
temperature_data_filtered = temperature_data[['Date', 'Avg', 'Avg_Min', 'Avg_Max']]

# 6. 결과를 CSV 파일로 저장합니다.
temperature_data_filtered.to_csv('./data/TSdata/Processed/temperature_data_filtered.csv', index=False)

print('temperature_data_filtered.csv 파일로 저장되었습니다.')"""
