### 설치 필요  : pip install selenium
# 동적 웹페이지 처리를 위한 라이브러리
from selenium import webdriver

# 웹페이지 내에 데이터 추출을 위한 라이브러리
from selenium.webdriver.common.by import By

# 시간 라이브러리 추가
import time

### 설치 필요  : pip install selenium
# 동적 웹페이지 처리를 위한 라이브러리
from selenium import webdriver

# 웹페이지 내에 데이터 추출을 위한 라이브러리
from selenium.webdriver.common.by import By

# 시간 라이브러리 추가
import time

driver = webdriver.Chrome()

### url을 이용하여 페이지 접근
# - get() : 페이지에 접근 후 해당 html 코드 읽어 들이기
# - driver 객체가 모든 정보를 가지고 있음


#전기세
driver.get("https://www.chosun.com/nsearch/?query=%ED%95%9C%EC%9A%B0&page=1&siteid=&sort=2&date_period=direct"
           "&date_start=20140101&date_end=20231231&writer=&field=&emd_word=&expt_word=&opt_chk=false&app_check=0"
           "&website=www,chosun&category=")

try:
    # 크롬 브라우저 띄우기
    driver = webdriver.Chrome()

    driver.implicitly_wait(10)  # 10초로 설정, 필요에 따라 조절

    # 페이지 수 조절 (원하는 페이지 수로 변경 가능)
    total_pages = 1

    for page in range(1, total_pages + 1):
        # url을 이용하여 페이지 접근
        url = "https://www.chosun.com/nsearch/?query=%ED%95%9C%EC%9A%B0&page={page}&siteid=&sort=2&date_period=direct"\
              "&date_start=20140101&date_end=20231231&writer=&field=&emd_word=&expt_word=&opt_chk=false&app_check=0"\
              "&website=www,chosun&category="
        driver.get(url)

        # 영화제목 위치 경로
        news_cho_path = "#main > div.search-feed > div > div > " \
                        "div.story-card.story-card--art-left.\|.flex.flex--wrap.box--hidden-sm > " \
                        "div.story-card-right.\|.grid__col--sm-8.grid__col--md-8.grid__col--lg-8.box--pad-left-xs > " \
                        "div.story-card__headline-container.\|.box--margin-bottom-xs > div > a "

        # 현재 크롬브라우저에 보이는 영화제목 모두 추출
        news_cho_elements = driver.find_elements(By.CSS_SELECTOR, news_cho_path)

        # 수집된 데이터를 파일에 저장
        with open("./data/news_cho_price.txt", "a", encoding="UTF-8") as f:
            for i, title_element in enumerate(news_cho_elements):
                title = title_element.text.strip()
                print(f"No[{i + 1}] / title[{title}] on page[{page}]")
                f.write(f"No[{i + 1}] / title[{title}] on page[{page}]\n")

                # 페이지 로딩 및 코드 읽어들이는 시간을 벌어주기
                time.sleep(1)

        ### 영화 한편에 대한 정보 수집이 끝나면 다시 메인으로 이동
        # - execute_script() : 자바스크립트 문법 처리 함수
        driver.execute_script("window.history.go(-2)")
        time.sleep(1)



except Exception as e:
    print(e)

finally:
    # 웹크롤링 처리가 모두 완료되면 driver 종료
    driver.quit()