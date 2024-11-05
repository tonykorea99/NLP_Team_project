import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from konlpy.tag import Okt
import matplotlib.font_manager as fm
import seaborn as sns

# 파일 경로
file1_path = 'C:\\LLM\\NLP_Team_Project\\25_9_categorized.csv'
file2_path = 'C:\\LLM\\NLP_Team_Project\\25_6_categorized.csv'

# 파일 읽기
data_csv = pd.read_csv(file1_path, encoding='utf-8')
data_excel = pd.read_excel(file2_path, engine='openpyxl')

# 형태소 분석기 초기화
okt = Okt()

# 추가적인 불용어 목록
stopwords = ['그', '수', '있다', '하다', '되다', '말하다', '없다', '그렇다', '같다', '이렇다', '저렇다', '하다']

# 전처리 함수: 품사 태깅 후 특정 품사와 불용어를 걸러내는 함수
def preprocess_text(text, pos_tags=['Noun', 'Verb']):
    tagged_words = okt.pos(text)
    meaningful_words = [
        word for word, pos in tagged_words 
        if pos in pos_tags and word not in stopwords and len(word) > 1
    ]
    return ' '.join(meaningful_words)

# 전처리 적용
data_csv['processed_text'] = data_csv['test'].apply(lambda x: preprocess_text(x, pos_tags=['Noun']))
data_excel['processed_text'] = data_excel['test'].apply(lambda x: preprocess_text(x, pos_tags=['Noun']))

# 전체 텍스트를 하나의 문자열로 결합
all_text = ' '.join(data_csv['processed_text'].tolist() + data_excel['processed_text'].tolist())
all_words = all_text.split()

# 키워드 빈도 계산
word_freq = Counter(all_words)

# 한글 폰트 경로 설정 (Malgun Gothic 폰트를 사용)
font_path = 'C:\\Windows\\Fonts\\malgun.ttf'

# 전체 키워드 빈도 워드클라우드 시각화
wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Overall Keyword Frequency")
plt.show()

# 6월과 9월 데이터의 카테고리별 문제 수 계산
june_category_counts = data_excel['category'].value_counts()
sept_category_counts = data_csv['category'].value_counts()

# 6월과 9월 카테고리 문제 수를 하나의 데이터프레임으로 결합
category_trends = pd.DataFrame({
    'June': june_category_counts,
    'September': sept_category_counts
}).fillna(0)  # 결측값은 0으로 대체

# 스택형 막대 그래프 시각화
category_trends.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon'])
plt.xlabel('Category', fontproperties=fm.FontProperties(fname=font_path))
plt.ylabel('Number of Problems')
plt.title('Number of Problems by Category in June and September (Stacked)')
plt.xticks(rotation=45, fontproperties=fm.FontProperties(fname=font_path))
plt.legend(title='Month')
plt.show()

# 변화량 데이터 출력
print(category_trends)
