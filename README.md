# 데이터 분석가를 위한 데이터 엔지니어링 기초 다지기 with Python

이 프로젝트는 데이터 분석가가 데이터 엔지니어링의 기초를 다질 수 있도록 구성된 실습 튜토리얼입니다. Python을 기반으로 데이터 수집, 처리, 저장, 분석의 전체 파이프라인을 학습할 수 있습니다.

## 표지

![](/img/book_cover.png)

## 📚 챕터별 학습 내용

### Ch01: Python 기초 및 환경 설정
- **목표**: Python 개발 환경 구축 및 기본 문법 학습
- **주요 내용**:
  - Python 환경 설정 및 가상환경 구성
  - 기본 문법 및 데이터 타입
  - Jupyter Notebook 활용법
- **파일**: `test.ipynb`, `app.py`

### Ch02: 데이터 처리 기초
- **목표**: 데이터 처리의 기본 개념과 방법론 학습
- **주요 내용**:
  - 데이터 처리 파이프라인 이해
  - ETL(Extract, Transform, Load) 개념
  - 데이터 품질 관리

### Ch03: 병렬 처리 및 멀티스레딩
- **목표**: Python에서의 병렬 처리 기법 학습
- **주요 내용**:
  - 멀티스레딩과 멀티프로세싱
  - ThreadPoolExecutor 활용
  - ProcessPoolExecutor 활용
  - 성능 최적화 기법
- **파일**: `multiprocess_sample.ipynb`, `thread_example.ipynb`

### Ch04: 데이터 수집 및 크롤링
- **목표**: 다양한 소스에서 데이터를 수집하는 방법 학습
- **주요 내용**:
  - **CSV 처리**: `csv_sample/` - CSV 파일 읽기/쓰기
  - **JSON 처리**: `json_sample/` - JSON 데이터 파싱 및 생성
  - **Excel 처리**: `excel_sample/` - Excel 파일 읽기/쓰기
  - **데이터베이스 연동**: `db_sample/` - DB 연결 및 쿼리
  - **웹 크롤링**: `webcrawling/` - 웹 스크래핑 기법

### Ch05: 데이터베이스 및 SQL
- **목표**: 데이터베이스 설계 및 SQL 활용법 학습
- **주요 내용**:
  - MySQL 데이터베이스 설계
  - 고급 SQL 쿼리 작성
  - 서브쿼리 및 윈도우 함수
  - 데이터 마트 구축
- **파일**: 
    - `mysqlsampledatabase.sql` - 샘플 데이터베이스
    - `create_datamart.sql` - 데이터 마트 생성
    - `subquery.sql` - 서브쿼리 예제
    - `windowfunctions.sql` - 윈도우 함수 예제
    - `apt_ingestion.py` - 아파트 데이터 수집

### Ch06: 데이터 변환 및 전처리
- **목표**: 데이터 정제, 변환, 시각화 기법 학습
- **주요 내용**:
  - **NumPy 벡터화**: `numpy_vectorization.ipynb` - 고성능 배열 연산
  - **데이터프레임 변환**: `dataframe_transformation.ipynb` - Pandas 활용
  - **날짜/시간 처리**: `datetime_transformation.ipynb` - 시간 데이터 변환
  - **결측값 처리**: `missing_values.ipynb` - 결측값 대체 및 시각화
  - **정규표현식**: `regular_expressions.ipynb` - 텍스트 패턴 매칭
  - **문자열 변환**: `string_transformation.ipynb` - 텍스트 데이터 처리
- **데이터**: `apartment_sales_data.csv` - 아파트 매매 데이터

### Ch07: 데이터베이스 연동 및 로그 분석
- **목표**: 다양한 데이터베이스와의 연동 및 로그 데이터 분석
- **주요 내용**:
  - **MongoDB 연동**: `mongodb_sample.ipynb` - NoSQL 데이터베이스 활용
  - **MySQL 연동**: `mysqlconnector_sqlalchemy.ipynb` - SQLAlchemy ORM
  - **로그 분석**: 
    - `logfile_generator.ipynb` - 로그 파일 생성
    - `logfile_analysis.ipynb` - 로그 데이터 분석
    - `upload_logfile.ipynb` - 로그 파일 업로드
- **데이터**: `data/` - 로그 파일 및 샘플 데이터

### Ch08: ETL 파이프라인 및 벡터 검색
- **목표**: 완전한 ETL 파이프라인 구축 및 AI 기반 이미지 검색 시스템 개발
- **주요 내용**:
  - **ETL 파이프라인**: 
    - `etl_book_pipeline.py` - 도서 데이터 수집 및 벡터화
    - `extract.py` - 웹 크롤링을 통한 데이터 추출
    - `transform.py` - 데이터 변환 및 정제
    - `image_etl.py` - 이미지 데이터 처리 및 벡터화
  - **벡터 데이터베이스**: 
    - `duckdb_milvus_check.ipynb` - Milvus 벡터 DB 연동
    - `book_data.duckdb` - DuckDB 메타데이터 저장
  - **웹 애플리케이션**: 
    - `main.py` - FastAPI 기반 웹 서버
    - `app.py` - Streamlit 대시보드
    - `image_search.py` - 이미지 검색 기능
    - `get_book_data.py` - 도서 데이터 조회
  - **프론트엔드**: `templates/` - HTML 템플릿
- **데이터**: `data/` - 도서 이미지 및 메타데이터

## 기술 스택

### 프로그래밍 언어
- **Python 3.9+**: 메인 프로그래밍 언어
- **Jupyter Notebook**: 데이터 분석 및 실습

### 데이터 처리 및 분석
- **NumPy**: 수치 계산 및 배열 연산
- **Pandas**: 데이터 조작 및 분석
- **Polars**: 고성능 데이터 처리
- **PyArrow**: 메모리 효율적인 데이터 처리
- **Scikit-learn**: 머신러닝 및 데이터 전처리
- **Missingno**: 결측값 시각화 및 분석

### 데이터 시각화
- **Matplotlib**: 기본 차트 및 그래프
- **Seaborn**: 통계 데이터 시각화

### 데이터베이스 및 저장소
- **DuckDB**: 임베딩 데이터베이스
- **MongoDB**: NoSQL 데이터베이스
- **MySQL**: 관계형 데이터베이스
- **SQLAlchemy**: ORM 및 데이터베이스 추상화
- **PyMongo**: MongoDB Python 드라이버

### AI/ML 및 벡터 검색
- **SentenceTransformers**: 텍스트 및 이미지 임베딩
- **PyMilvus 2.6.0**: 벡터 데이터베이스 클라이언트

### 웹 프레임워크 및 UI
- **FastAPI**: 고성능 API 서버
- **Streamlit**: 데이터 대시보드
- **Jinja2**: HTML 템플릿 엔진
- **Uvicorn**: ASGI 서버

### 데이터 수집 및 처리
- **BeautifulSoup**: 웹 스크래핑
- **Requests**: HTTP 클라이언트
- **Advertools**: 웹 크롤링 도구
- **OpenPyXL**: Excel 파일 처리
- **XLrd**: Excel 파일 읽기

### 유틸리티
- **Python-multipart**: 파일 업로드 처리

## 🚀 시작하기

### 1. 환경 설정
- [uv 공식 문서](https://github.com/astral-sh/uv)

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# 저장소 클론
git clone [repository-url]
cd de_etl_book_tutorial

# 가상환경 생성 및 활성화
uv venv --python 3.11
source .venv/bin/activate  # Linux/Mac


# 의존성 설치
uv pip install -r requirements.txt
```

### 2. 데이터베이스 설정
```bash
# Milvus 벡터 데이터베이스 시작
bash standalone_embed.sh start

# MySQL 샘플 데이터베이스 생성
mysql -u root -p < ch05/mysqlsampledatabase.sql
```

### 3. ETL 파이프라인 실행
```bash
# 도서 데이터 수집 및 벡터화
cd ch08
python etl_book_pipeline.py
```

### 4. PyTorch 설치
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
``` 

### 5. 웹 애플리케이션 실행
```bash
# FastAPI 서버 시작
uvicorn main:app --reload --port 8000

# Streamlit 대시보드 시작
streamlit run app.py --server.port 8501
```

## 📊 학습 로드맵

1. **기초 단계** (Ch01-Ch03): Python 기초 및 병렬 처리
2. **데이터 수집 단계** (Ch04): 다양한 소스에서 데이터 수집
3. **데이터 처리 단계** (Ch05-Ch06): 데이터베이스 및 전처리
4. **고급 단계** (Ch07-Ch08): 실무 프로젝트 및 AI 시스템 구축

## 🔍 주요 프로젝트

### 도서 이미지 검색 시스템 (Ch08)
- **기능**: AI 기반 이미지 유사도 검색
- **기술**: CLIP 모델, Milvus, FastAPI, Streamlit
- **데이터**: Books to Scrape 웹사이트 크롤링

### 로그 분석 시스템 (Ch07)
- **기능**: 로그 데이터 수집, 저장, 분석
- **기술**: MongoDB, MySQL, Pandas
- **데이터**: 생성된 로그 파일

### 아파트 매매 데이터 분석 (Ch05-Ch06)
- **기능**: 부동산 데이터 수집 및 분석
- **기술**: SQL, Pandas, 데이터 시각화
- **데이터**: 실제 아파트 매매 데이터

## 📦 의존성 패키지 상세

### 핵심 데이터 처리
- `numpy`: 수치 계산 및 배열 연산
- `pandas`: 데이터 조작 및 분석
- `polars`: 고성능 데이터 처리
- `pyarrow`: 메모리 효율적인 데이터 처리

### 데이터 시각화
- `matplotlib`: 기본 차트 및 그래프
- `seaborn`: 통계 데이터 시각화

### 데이터베이스 연동
- `duckdb`: 임베딩 데이터베이스
- `pymongo`: MongoDB Python 드라이버
- `mysql-connector-python`: MySQL 연결
- `sqlalchemy`: ORM 및 데이터베이스 추상화

### AI/ML 및 벡터 검색
- `sentence-transformers`: 텍스트 및 이미지 임베딩
- `pymilvus==2.6.0`: 벡터 데이터베이스 클라이언트
- `scikit-learn`: 머신러닝 및 데이터 전처리

### 웹 프레임워크
- `fastapi[all]`: 고성능 API 서버 (모든 의존성 포함)
- `streamlit`: 데이터 대시보드
- `jinja2`: HTML 템플릿 엔진
- `uvicorn`: ASGI 서버

### 데이터 수집 및 처리
- `beautifulsoup4`: 웹 스크래핑
- `requests`: HTTP 클라이언트
- `advertools`: 웹 크롤링 도구
- `openpyxl`: Excel 파일 처리
- `xlrd`: Excel 파일 읽기

### 개발 도구
- `jupyterlab`: Jupyter Notebook 환경
- `missingno`: 결측값 시각화
- `python-multipart`: 파일 업로드 처리

## 📝 참고 자료

- **PDF**: `데이터 분석가를 위한 데이터 엔지니어링 기초 다지기 with Python_250828.pdf`
- **SQL 스크립트**: `get_data.sql` - 데이터 조회 예제
- **설정 파일**: `embedEtcd.yaml`, `user.yaml` - 시스템 설정

## 🤝 기여하기

이 프로젝트는 학습 목적으로 제작되었습니다. 개선사항이나 버그 리포트는 이슈로 등록해 주세요.

## 📄 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.