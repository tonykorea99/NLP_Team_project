import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from konlpy.tag import Okt
import os
import glob
import chardet

# matplotlib의 폰트를 맑은 고딕으로 설정
plt.rc('font', family='Malgun Gothic')  # 'Malgun Gothic'은 맑은 고딕 폰트 이름입니다.

# 형태소 분석 및 불용어 처리 설정
okt = Okt()
# 불용어 리스트에 자주 등장하는 불필요한 단어들 추가
stopwords = ['그', '수', '있다', '하다', '되다', '없다', '같다', '이렇다', '저렇다', '하다', '하는', '합니다', 
             '있습니다', '그리고', '입니다', '것', '정도', '이번', '위해', '관련', '때문', '대해', '이후', '경우', 
             '대한', '우리', '사용', '더', '또', '합니다', '대한', '해야', '하기']

# 명사 추출을 위한 전처리 함수 정의
def preprocess_text(text):
    tagged_words = okt.pos(text)
    # 명사(Noun)만 추출하고 불용어 제거
    meaningful_words = [
        word for word, pos in tagged_words 
        if pos == 'Noun' and word not in stopwords and len(word) > 1
    ]
    return ' '.join(meaningful_words)

# 파일 인코딩 감지 함수 정의
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# 로컬 파일 경로
data_dir = r'C:\Users\1\Desktop\NLP\NLP_Team_project\NLP_Data'
yearly_data = {}

# 파일 경로에서 각 연도별로 데이터를 병합
for filepath in glob.glob(os.path.join(data_dir, '*.CSV')):  # 모든 .CSV 파일을 불러옴
    filename = os.path.basename(filepath)
    year = int(filename.split('_')[0])  # 파일명에서 연도 추출
    
    if year not in yearly_data:
        yearly_data[year] = pd.DataFrame()
    
    # 파일 인코딩을 감지하여 데이터프레임으로 로드
    encoding = detect_encoding(filepath)
    data = pd.read_csv(filepath, encoding=encoding)
    
    if 'test' in data.columns:  # 'test' 열이 있는 경우만 처리
        data['processed_text'] = data['test'].apply(lambda x: preprocess_text(str(x)))
        yearly_data[year] = pd.concat([yearly_data[year], data], ignore_index=True)

# 워드 클라우드 생성 함수
def generate_wordcloud(word_freq, title):
    if word_freq:  # 빈 데이터 확인
        wordcloud = WordCloud(font_path='C:\\Windows\\Fonts\\malgun.ttf', width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()
    else:
        print(f"{title} - 생성할 단어가 없어 워드 클라우드를 건너뜁니다.")

# 연도별 키워드 워드 클라우드 생성
for year, data in yearly_data.items():
    if 'processed_text' in data.columns and not data['processed_text'].empty:
        all_words = ' '.join(data['processed_text'].tolist()).split()
        word_freq = Counter(all_words)
        generate_wordcloud(word_freq, f"{year}년도 키워드 빈도수")

# 전체 데이터 통합 후 워드 클라우드 생성
all_text = ' '.join(' '.join(data['processed_text'].tolist()) for data in yearly_data.values() if 'processed_text' in data.columns and not data['processed_text'].empty)
all_words = all_text.split()
word_freq = Counter(all_words)
generate_wordcloud(word_freq, "2021-2025 전체 키워드 빈도수")

# 카테고리 트렌드 분석
category_trends = pd.DataFrame()

for year, data in yearly_data.items():
    if 'category' in data.columns:
        category_counts = data['category'].value_counts()
        category_counts.name = year
        category_trends = pd.concat([category_trends, category_counts], axis=1)

category_trends = category_trends.fillna(0)  # 결측값은 0으로 대체

# 트렌드 변화 시각화
category_trends.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.xlabel('Category')
plt.ylabel('Number of Problems')
plt.title('Yearly Category Trends (2021-2025)')
plt.xticks(rotation=45)
plt.legend(title='Year')
plt.show()
