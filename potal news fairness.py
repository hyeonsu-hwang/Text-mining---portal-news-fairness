## 카카오주식 관련 기사 크롤링
import requests
from bs4 import BeautifulSoup

for n in range(1, 4):
    raw = requests.get("https://search.daum.net/search?w=news&q=카카오주식&p="+str(n))
    html = BeautifulSoup(raw.text, 'html.parser')

    articles = html.select("div.wrap_cont")

    for ar in articles:
        title = ar.select_one("a.f_link_b").text
        summary = ar.select_one("p.f_eb.desc").text

        print(title)
        print(summary)
        # 기사별로 구분을 위해 구분선 삽입 
        print("="*50)
        

## okt 형태소 분석기
from konlpy.tag import Okt
from collections import Counter

f = open('daum_dataframe.csv','r',encoding='utf-8')
news = f.read()

okt = Okt()
noun = okt.nouns(news)
count = Counter(noun)

noun_list = count.most_common(100)
for v in noun_list:
    print(v)

for i,v in enumerate(noun):
    if len(v)<2:
        noun.pop(i)
count = Counter(noun)

noun_list = count.most_common(100)
for v in noun_list:
    print(v)

with open("noun_list.txt",'w',encoding='utf-8') as f:
    for v in noun_list:
        f.write(" ".join(map(str,v)))
        f.write("\n")

import sys
from wordcloud import WordCloud

filename = sys.argv[1]
f = open('noun_list.txt','r',encoding='utf-8')
news = f.read()


wc = WordCloud(font_path='c:\windows\Fonts\H2PORL.TTF',
              background_color="white",
              width=1000,
              height=1000,
              max_words=100,
              max_font_size=300)

wc.generate_from_frequencies(dict(noun_list))

wc.to_file('wordcloud_news.png')

##
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import re
import sys
import time, random

header = { 
    "User-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36",      
}
   
def get_news(n_url):
    news_detail = []
    print(n_url)
    breq = requests.get(n_url, headers=header)
    bsoup = BeautifulSoup(breq.content, 'html.parser')

    # 제목 파싱
    title = bsoup.select('h3#articleTitle')[0].text
    news_detail.append(title)

    # 날짜
    pdate = bsoup.select('.t11')[0].get_text()[:11]
    news_detail.append(pdate)

    # news text
    _text = bsoup.select('#articleBodyContents')[0].get_text().replace('\n', " ")
    btext = _text.replace("// flash 오류를 우회하기 위한 함수 추가 function _flash_removeCallback() {}", "")
    news_detail.append(btext.strip())

    # 신문사
    pcompany = bsoup.select('#footer address')[0].a.get_text()
    news_detail.append(pcompany)

    return news_detail

columns = ['날짜','신문사', 'title','내용']
df = pd.DataFrame(columns=columns)

query = '카카오 주가'   # url 인코딩 에러는 encoding parse.quote(query) 
s_date = "2020.11.01" 
e_date = "2020.11.11" 
s_from = s_date.replace(".","") 
e_to = e_date.replace(".","") 
page = 1 

while True:
    time.sleep(random.sample(range(3), 1)[0])
    print(page) 
    url = "https://search.naver.com/search.naver?where=news&query=" + query + "&sort=1&field=1&ds=" + s_date + "&de=" + e_date +\
    "&nso=so%3Ar%2Cp%3Afrom" + s_from + "to" + e_to + "%2Ca%3A&start=" + str(page) 
    header = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36' 
    }
    req = requests.get(url,headers=header) 
    print(url) 
    cont = req.content 
    soup = BeautifulSoup(cont, 'html.parser') 

    if soup.findAll("a",{"class":"info"}) == [] :
        break 
    for urls in soup.findAll("a",{"class":"info"}): 
        try : 
            if urls.attrs["href"].startswith("https://news.naver.com"): 
                print(urls.attrs["href"]) 
                news_detail = get_news(urls.attrs["href"]) 
                df=df.append(pd.DataFrame([[news_detail[1], news_detail[3], news_detail[0], news_detail[2]]],columns=columns))
        except Exception as e: 
            print(e)  
            continue 
    page += 10  
    
    

#######################################################


ndf = df.reset_index()
my_title_df = ndf['title']

titles = list(my_title_df['title'])

###################### 감성 사전 불러오기 ######################
import codecs 

positive = [] 
negative = [] 
posneg = [] 

pos = codecs.open("./positive_words_self.txt", 'rb', encoding='UTF-8') 

while True: 
    line = pos.readline() 
    line = line.replace('\n', '') 
    positive.append(line) 
    posneg.append(line) 
    
    if not line: break 
pos.close()



neg = codecs.open("./negative_words_self.txt", 'rb', encoding='UTF-8') 

while True: 
    line = neg.readline() 
    line = line.replace('\n', '') 
    negative.append(line) 
    posneg.append(line) 
    
    if not line: break 
neg.close()
####################################### 라벨 붙이기 ##################################
# label 정보를 담을 빈 label 리스트 생성

label = []
for i in range(titles): 
  # re.sub을 통해서 기사 제목에서 특수문자 제거
  clean_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》]', '', titles[i]) 

  for j in range(len(posneg)): 
    posflag = False 
    negflag = False 
    if j < (len(positive)-1): 
      if clean_title.find(posneg[j]) != -1: 
        posflag = True 
        print(j, "positive?","테스트 : ",clean_title.find(posneg[j]),"비교단어 : ", posneg[j], "인덱스 : ", j, clean_title) 
        break 
    if j > (len(positive)-2): 
      if clean_title.find(posneg[j]) != -1: 
        negflag = True 
        print(j, "negative?","테스트 : ",clean_title.find(posneg[j]),"비교단어 : ", posneg[j], "인덱스 : ", j, clean_title) 
        break 
  if posflag == True: 
    label.append(1)
  elif negflag == True: 
    label.append(-1)
  elif negflag == False and posflag == False: 
    label.append(0)
  
# 데이터를 넣어준 label 리스트를 my_title_df에 추가
my_title_df['label'] = label
my_title_df



###
###################### 감성 사전 불러오기 ######################
import codecs 

positive = [] 
negative = [] 
posneg = [] 

pos = codecs.open("./positive_words.txt", 'rb', encoding='UTF-8') 

while True: 
    line = pos.readline() 
    line = line.replace('\n', '') 
    positive.append(line) 
    posneg.append(line) 
    
    if not line: break 
pos.close()



neg = codecs.open("./negative_words.txt", 'rb', encoding='UTF-8') 

while True: 
    line = neg.readline() 
    line = line.replace('\n', '') 
    negative.append(line) 
    posneg.append(line) 
    
    if not line: break 
neg.close()

# positive 
# negative
# posneg  애네 끝에 비어있는 행 제거 해줘야함

##################################### 네이버 뉴스 (주가 train data)
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import re
import sys
import time, random

header = { 
    "User-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36",      
}
   
def get_news(n_url):
    news_detail = []
    print(n_url)
    breq = requests.get(n_url, headers=header)
    bsoup = BeautifulSoup(breq.content, 'html.parser')

    # 제목 파싱
    title = bsoup.select('h3#articleTitle')[0].text
    news_detail.append(title)

    return news_detail

columns = ['title']
df = pd.DataFrame(columns=columns)

query = '주가'   # url 인코딩 에러는 encoding parse.quote(query) 
s_date = "2020.10.01" 
e_date = "2020.12.11" 
s_from = s_date.replace(".","") 
e_to = e_date.replace(".","") 
page = 1 

while True:
    time.sleep(random.sample(range(3), 1)[0])
    print(page) 
    url = "https://search.naver.com/search.naver?where=news&query=" + query + "&sort=1&field=1&ds=" + s_date + "&de=" + e_date +\
    "&nso=so%3Ar%2Cp%3Afrom" + s_from + "to" + e_to + "%2Ca%3A&start=" + str(page) 
    header = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36' 
    }
    req = requests.get(url,headers=header) 
    print(url) 
    cont = req.content 
    soup = BeautifulSoup(cont, 'html.parser') 

    if soup.findAll("a",{"class":"info"}) == [] :
        break 
    for urls in soup.findAll("a",{"class":"info"}): 
        try : 
            if urls.attrs["href"].startswith("https://news.naver.com"): 
                print(urls.attrs["href"]) 
                news_detail = get_news(urls.attrs["href"]) 
                df=df.append(pd.DataFrame([news_detail[0]],columns=columns))
        except Exception as e: 
            print(e)  
            continue 
    page += 10  
    



####################################### 라벨 붙이기 ##################################
# label 정보를 담을 빈 label 리스트 생성


ndf = df.reset_index()
my_title_df = pd.DataFrame(ndf['title'])

titles = list(my_title_df['title'])




label = []
for i in range(len(titles)): 
  # re.sub을 통해서 기사 제목에서 특수문자 제거
  clean_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》]', '', titles[i]) 

  for j in range(len(posneg)): 
    posflag = False 
    negflag = False 
    if j < (len(positive)-1): 
      if clean_title.find(posneg[j]) != -1: 
        posflag = True 
        print(j, "positive?","테스트 : ",clean_title.find(posneg[j]),"비교단어 : ", posneg[j], "인덱스 : ", j, clean_title) 
        break 
    if j > (len(positive)-2): 
      if clean_title.find(posneg[j]) != -1: 
        negflag = True 
        print(j, "negative?","테스트 : ",clean_title.find(posneg[j]),"비교단어 : ", posneg[j], "인덱스 : ", j, clean_title) 
        break 
  if posflag == True: 
    label.append(1)
  elif negflag == True: 
    label.append(-1)
  elif negflag == False and posflag == False: 
    label.append(0)
  
# 데이터를 넣어준 label 리스트를 my_title_df에 추가
my_title_df['label'] = label
my_title_df

# csv 파일로 저
my_title_df.to_csv('train_data.csv', sep=',', na_rep='NaN', encoding='utf-8')


################## 네이버 뉴스 (카카오주가 test data)
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import re
import sys
import time, random

header = { 
    "User-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36",      
}
   
def get_news(n_url):
    news_detail = []
    print(n_url)
    breq = requests.get(n_url, headers=header)
    bsoup = BeautifulSoup(breq.content, 'html.parser')

    # 제목 파싱
    title = bsoup.select('h3#articleTitle')[0].text
    news_detail.append(title)

    return news_detail

columns = ['title']
df = pd.DataFrame(columns=columns)

query = '카카오 주가'   # url 인코딩 에러는 encoding parse.quote(query) 
s_date = "2020.10.01" 
e_date = "2020.12.11" 
s_from = s_date.replace(".","") 
e_to = e_date.replace(".","") 
page = 1 

while True:
    time.sleep(random.sample(range(3), 1)[0])
    print(page) 
    url = "https://search.naver.com/search.naver?where=news&query=" + query + "&sort=1&field=1&ds=" + s_date + "&de=" + e_date +\
    "&nso=so%3Ar%2Cp%3Afrom" + s_from + "to" + e_to + "%2Ca%3A&start=" + str(page) 
    header = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36' 
    }
    req = requests.get(url,headers=header) 
    print(url) 
    cont = req.content 
    soup = BeautifulSoup(cont, 'html.parser') 

    if soup.findAll("a",{"class":"info"}) == [] :
        break 
    for urls in soup.findAll("a",{"class":"info"}): 
        try : 
            if urls.attrs["href"].startswith("https://news.naver.com"): 
                print(urls.attrs["href"]) 
                news_detail = get_news(urls.attrs["href"]) 
                df=df.append(pd.DataFrame([news_detail[0]],columns=columns))
        except Exception as e: 
            print(e)  
            continue 
    page += 10  
    



####################################### 라벨 붙이기 ##################################
# label 정보를 담을 빈 label 리스트 생성


ndf = df.reset_index()
my_title_df = pd.DataFrame(ndf['title'])

titles = list(my_title_df['title'])




label = []
for i in range(len(titles)): 
  # re.sub을 통해서 기사 제목에서 특수문자 제거
  clean_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》]', '', titles[i]) 

  for j in range(len(posneg)): 
    posflag = False 
    negflag = False 
    if j < (len(positive)-1): 
      if clean_title.find(posneg[j]) != -1: 
        posflag = True 
        print(j, "positive?","테스트 : ",clean_title.find(posneg[j]),"비교단어 : ", posneg[j], "인덱스 : ", j, clean_title) 
        break 
    if j > (len(positive)-2): 
      if clean_title.find(posneg[j]) != -1: 
        negflag = True 
        print(j, "negative?","테스트 : ",clean_title.find(posneg[j]),"비교단어 : ", posneg[j], "인덱스 : ", j, clean_title) 
        break 
  if posflag == True: 
    label.append(1)
  elif negflag == True: 
    label.append(-1)
  elif negflag == False and posflag == False: 
    label.append(0)
  
# 데이터를 넣어준 label 리스트를 my_title_df에 추가
my_title_df['label'] = label
my_title_df

# csv 파일로 저
my_title_df.to_csv('test_data.csv', sep=',', na_rep='NaN', encoding='utf-8')







########### 데이터 분석 ###############


import pandas as pd 
train_data = pd.read_csv("./train_data.csv") 
test_data = pd.read_csv("./test_data.csv")

%matplotlib inline 
import matplotlib.pyplot as plt

train_data['label'].value_counts().plot(kind='bar')

test_data['label'].value_counts().plot(kind='bar')

print(train_data.groupby('label').size().reset_index(name='count'))
print(test_data.groupby('label').size().reset_index(name='count'))


# 모델을 만들기 위한 전처리
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# 토큰화
import konlpy 
from konlpy.tag import Okt 
okt = Okt() 
X_train = [] 
for sentence in train_data['title']: 
    temp_X = [] 
    temp_X = okt.morphs(sentence, stem=True) # 토큰화 
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 
    X_train.append(temp_X) 
    
X_test = [] 
for sentence in test_data['title']: 
    temp_X = [] 
    temp_X = okt.morphs(sentence, stem=True) # 토큰화 
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 
    X_test.append(temp_X)

#토큰화 한 단어 컴퓨터 인식 (정수 인코딩)

import tensorflow as tf
import keras

from keras.preprocessing.text import Tokenizer 
max_words = 35000 
tokenizer = Tokenizer(num_words = max_words) 
tokenizer.fit_on_texts(X_train) 
X_train = tokenizer.texts_to_sequences(X_train) 
X_test = tokenizer.texts_to_sequences(X_test)

# y값으로 들어갈 label -1,0,1을 컴퓨터가 인식하도록 one-hot encording
import numpy as np 

y_train = [] 
y_test = [] 


for i in range(len(train_data['label'])): 
    if train_data['label'].iloc[i] == 1: 
        y_train.append([0, 0, 1]) 
    elif train_data['label'].iloc[i] == 0: 
        y_train.append([0, 1, 0]) 
    elif train_data['label'].iloc[i] == -1: 
        y_train.append([1, 0, 0]) 
        
for i in range(len(test_data['label'])): 
    if test_data['label'].iloc[i] == 1: 
        y_test.append([0, 0, 1]) 
    elif test_data['label'].iloc[i] == 0: 
        y_test.append([0, 1, 0]) 
    elif test_data['label'].iloc[i] == -1: 
        y_test.append([1, 0, 0]) 
        
        
y_train = np.array(y_train) 
y_test = np.array(y_test)


#모델 만들기
from keras.layers import Embedding, Dense, LSTM 
from keras.models import Sequential 
from keras.preprocessing.sequence import pad_sequences 
max_len = 20 # 전체 데이터의 길이를 20로 맞춘다 

X_train = pad_sequences(X_train, maxlen=max_len) 
X_test = pad_sequences(X_test, maxlen=max_len)

model = Sequential() 
model.add(Embedding(max_words, 100)) 
model.add(LSTM(128)) 
model.add(Dense(3, activation='softmax')) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.1)

print("\n 테스트 정확도 : {:.2f}%".format(model.evaluate(X_test, y_test)[1]*100)) 


## 기사 크롤링
import requests
from bs4 import BeautifulSoup

for n in range(1, 50):
    raw = requests.get("https://search.daum.net/search?w=news&q=카카오주가&p="+str(n))
    html = BeautifulSoup(raw.text, 'html.parser')

    articles = html.select("div.wrap_cont")

    for ar in articles:
        title = ar.select_one("a.f_link_b").text
        summary = ar.select_one("p.f_eb.desc").text

        print(title)
        print(summary)
        # 기사별로 구분을 위해 구분선 삽입 
        print("="*50)

import requests
from bs4 import BeautifulSoup

#############################################
# 추가1

# openpyxl 패키지를 불러와서 새 워크북을 만듭니다.
# 헤더 부분을 채워줍니다.
import openpyxl
wb = openpyxl.Workbook()
sheet = wb.active
sheet.append(["title", "contents"])
#############################################

keyword = input("검색어를 입력해주세요: ")

# 반복1: 기사번호를 변경시키면서 데이터 수집을 반복하기
# 1 ~ 100까지 10단위로 반복(1, 11, ..., 91)
for n in range(1, 50):
    raw = requests.get("https://search.daum.net/search?w=news&q=카카오주식&p="+str(n))
    html = BeautifulSoup(raw.text, 'html.parser')

    articles = html.select("div.wrap_cont")

    for ar in articles:
        title = ar.select_one("a.f_link_b").text
        summary = ar.select_one("p.f_eb.desc").text

        print(title,summary)
        print("="*50)
        sheet.append([title, summary])
        # 기사별로 구분을 위해 구분선 삽입
   
   # 반복2: 기사에 대해서 반복하면 세부 정보 수집하기                
   # 리스트를 사용한 반복문으로 모든 기사에 대해서 제목/언론사 출력

       #############################################
       # 추가2
       
       # 수집한 제목, 언론사를 엑셀 파일에 써줍니다.

       #############################################

#############################################
# 추가3

# 작성한 워크북(엑셀파일)을 navertv.xlsx로 저장합니다.
wb.save("daum.xlsx")


xlsx = pd.read_excel("daum.xlsx")
xlsx.to_csv("daum.csv")


############### 각문서별 워드클라우드 코드
from konlpy.tag import Okt
from konlpy.corpus import kolaw
import nltk
okt = Okt()

f = open('daum_dataframe.csv','r',encoding='utf-8')
news = f.read()

okt.pos(news)

okt.nouns(news)

ko = nltk.Text(token_ko)
ko

ko.vocab()

stopword=['이','저','데','빅히트','등','주','하이닉스','삼성','테슬라','기자','셀트리온']

ko = [i for i in ko if i not in stopword]
ko

ko = nltk.Text(ko)

data = ko.vocab().most_common(300)
data

ko_vocab = ko.vocab()

token_ko = okt.nouns(news)
token_ko

from urllib import request
from wordcloud import ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import sys
from wordcloud import WordCloud

kakao_mask = np.array(Image.open("kakao_logo.jpg")) #이미지 파일 집어넣으시면 되요

#font_path는 로컬디스크C->Windows->font->안에있는 폰트 복사후 바탕화면에 붙여넣기 후 파일이름 확인하면됨

wordcloud = WordCloud(font_path='c:\windows\Fonts\H2PORL.TTF', 
                      stopwords = stopword,
                      background_color = 'white',
                      width = 1000, height = 800,max_font_size=300,mask = kakao_mask).generate_from_frequencies(dict(data))
plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()








