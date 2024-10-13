import os
import sys
import urllib.request
import pandas as pd
import json

client_id = "b7mNTHqkHbwLp7ObfA0s"
client_secret = "gZ923kFy_1"
encText = urllib.parse.quote("한우")
url = "https://openapi.naver.com/v1/search/news.json?query=" + encText # JSON 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
url+="&display=100&start=1000&sort=date"
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()

if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)

# JSON 데이터를 파이썬 딕셔너리로 변환
json_data = json.loads(response_body.decode('utf-8'))

# 'items' 항목에 있는 기사 데이터 추출
articles = json_data['items']

# pandas DataFrame으로 변환
df = pd.DataFrame(articles)

df['pubDate'] = pd.to_datetime(df['pubDate'], errors='coerce').dt.strftime('%Y-%m-%d')

# DataFrame을 CSV 파일로 저장
df.to_csv('naver_news_articles.csv', index=False, encoding='utf-8-sig')