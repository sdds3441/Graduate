import pandas as pd

# 1. CSV 파일을 읽어오면서 불필요한 메타데이터 행을 건너뜁니다.
file_path = 'data/TSdata/Orignal/extremum_20241010132919.csv'
rainfall_data = pd.read_csv(file_path, encoding='ISO-8859-1', skiprows=9, header=None)

# 2. 첫 번째 행은 필요 없는 데이터이므로 삭제하고, 마지막 빈 열도 제거합니다.
rainfall_data = rainfall_data.drop(0).drop(columns=[8])

# 3. 열 이름을 지정합니다.
rainfall_data.columns = [
    '지점번호', '지점명', '일시', '강수량(mm)', '일최대강수량(mm)',
    '일최대강수량일자', '1시간최대강수량(mm)', '1시간최대강수량일자'
]

# 4. 데이터에 포함된 공백이나 특수 문자를 제거합니다.
rainfall_data = rainfall_data.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

# 5. '지점명' 열의 한글 인코딩 문제를 해결합니다.
rainfall_data['지점명'] = rainfall_data['지점명'].apply(lambda x: x.encode('ISO-8859-1').decode('euc-kr'))

# 6. 필요한 열인 '일시'와 '강수량(mm)'만 선택하고, 열 이름을 영어로 변경합니다.
rainfall_data_filtered = rainfall_data[['일시', '강수량(mm)']].rename(columns={'일시': 'Date', '강수량(mm)': 'Rain'})

# 7. 결과를 출력하거나 원하는 파일로 저장합니다.
print(rainfall_data_filtered.head())

# 데이터 저장 예시 (csv 파일로 저장)
rainfall_data_filtered.to_csv('./data/TSdata/Processed/rainfall_data_filtered.csv', index=False)

print('rainfall_data_filtered.csv 파일로 저장되었습니다.')
