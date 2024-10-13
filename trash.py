import pandas as pd

# CSV 파일 로드
df = pd.read_csv('./data/TSdata/Processed/Whole_data.csv')

# 'Date' 열이 있다고 가정하고, 날짜 형식을 'YYYYMMDD'로 변환
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y%m%d')

# 변환된 데이터를 저장
df.to_csv('./data/Whole_data.csv', index=False, encoding='utf-8-sig')

print("날짜 형식 변환 완료")
