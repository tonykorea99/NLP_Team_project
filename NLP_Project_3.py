import pandas as pd
import os
import glob
from collections import Counter
from konlpy.tag import Okt
import chardet
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 형태소 분석 및 불용어 처리 설정
okt = Okt()
stopwords = ['그', '수', '있다', '하다', '되다', '없다', '같다', '이렇다', '저렇다', '합니다', '그리고', '입니다',
             '것', '정도', '이번', '위해', '관련', '때문', '대해', '이후', '경우', '대한', '우리', '사용', '더', '또']

def preprocess_text(text):
    tagged_words = okt.pos(text)
    meaningful_words = [
        word for word, pos in tagged_words 
        if pos == 'Noun' and word not in stopwords and len(word) > 1
    ]
    return ' '.join(meaningful_words)

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    print(f"Detected encoding for {file_path}: {result['encoding']}")
    return result['encoding']

# 데이터 로드 및 전처리
file_dir = r'C:\NLP_Project\NLP_Team_project\NLP_Data'  # 로컬 데이터 경로
all_files = glob.glob(os.path.join(file_dir, '*.csv'))
dataframes = []

for file_path in all_files:
    try:
        encoding = detect_encoding(file_path)
        data = pd.read_csv(file_path, encoding=encoding)
        if 'T' in data.columns and 'Q' in data.columns and 'C' in data.columns and 'A' in data.columns:
            data['processed_passage'] = data['T'].apply(lambda x: preprocess_text(str(x)))
            data['processed_question'] = data['Q'].apply(lambda x: preprocess_text(str(x)))
            dataframes.append(data)
        else:
            print(f"Warning: File {file_path} does not contain the required columns: 'T', 'Q', 'C', 'A'")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# 데이터 병합
if dataframes:
    data = pd.concat(dataframes, ignore_index=True)
else:
    raise ValueError("No valid dataframes to concatenate. Please check the input files.")

# 결측값 처리
if data['A'].isnull().any():
    print("Found NaN values in 'A' column. Dropping these rows.")
    data = data.dropna(subset=['A'])  # NaN 값 제거

data['A'] = data['A'].astype(float)  # 정답률을 float으로 변환

# 지문과 문제의 키워드 추출
vectorizer = CountVectorizer(max_features=1)
data['passage_keyword'] = data['processed_passage'].apply(lambda x: vectorizer.fit([x]).get_feature_names_out()[0] if x else '')
data['question_keyword'] = data['processed_question'].apply(lambda x: vectorizer.fit([x]).get_feature_names_out()[0] if x else '')

# 정답률 예측을 위한 데이터 준비
X = data['question_keyword']
y = data['A']

# 키워드를 벡터화
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# 새로운 테스트 데이터 예측 함수
def predict_accuracy(passage_text, question_text):
    processed_passage = preprocess_text(passage_text)
    processed_question = preprocess_text(question_text)
    passage_keyword = vectorizer.transform([processed_passage]).toarray() if processed_passage else None
    question_keyword = vectorizer.transform([processed_question]).toarray() if processed_question else None
    
    if question_keyword is not None:
        prediction = model.predict(question_keyword)[0]
    else:
        prediction = None
    return passage_keyword, question_keyword, prediction

# 사용자 입력을 통한 테스트
def user_test():
    print("지문과 문제를 입력하세요.")
    user_passage = input("지문: ")
    user_question = input("문제: ")
    passage_keyword, question_keyword, predicted_accuracy = predict_accuracy(user_passage, user_question)
    print(f"지문의 키워드: {passage_keyword}")
    print(f"문제의 키워드: {question_keyword}")
    print(f"예측된 정답률: {predicted_accuracy:.2f}")

# 사용자 테스트 실행
user_test()
