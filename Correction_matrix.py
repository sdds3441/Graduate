import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('./data/TSdata/Processed/whole_data.csv')

# 'Date' 열을 제외한 나머지 열로 상관계수 계산
#numeric_data = data.drop(columns=['Date'])  # 'Date' 열 제거
corr_matrix = data.corr()

# 상관계수 행렬 시각화 (seaborn 사용)
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
