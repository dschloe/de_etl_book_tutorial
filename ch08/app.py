import streamlit as st
import duckdb
import pandas as pd
import os

# --- 대시보드 설정 ---
st.set_page_config(
    page_title="DuckDB 도서 데이터 대시보드",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DuckDB 도서 데이터 분석 대시보드")
st.markdown("---")

# --- DuckDB 연결 및 데이터 로드 ---
DUCKDB_DB_FILE = "book_data.duckdb"

if not os.path.exists(DUCKDB_DB_FILE):
    st.error(f"'{DUCKDB_DB_FILE}' 파일이 존재하지 않습니다. 먼저 ETL 파이프라인을 실행하여 데이터베이스를 생성해주세요.")
    st.stop()

@st.cache_resource
def get_duckdb_connection():
    try:
        con = duckdb.connect(database=DUCKDB_DB_FILE, read_only=True)
        return con
    except Exception as e:
        st.error(f"DuckDB 연결 오류: {e}")
        st.stop()

con = get_duckdb_connection()

# --- 데이터 미리보기 ---
st.header("데이터 미리보기")
df = con.execute("SELECT * FROM book_metadata LIMIT 10").fetchdf()
st.dataframe(df)
st.markdown("---")

# --- 가격 분포 시각화 (Pandas + Streamlit) ---
st.header("가격 분포")
price_query = "SELECT price FROM book_metadata WHERE price IS NOT NULL;"
price_df = con.execute(price_query).fetchdf()

# Pandas의 'cut' 함수를 사용하여 가격을 20개의 구간으로 나눕니다.
# 각 구간을 문자열로 변환하여 'SchemaValidationError' 오류를 방지합니다.
price_df['price_bin'] = pd.cut(price_df['price'], bins=20)
binned_price_df = price_df.groupby('price_bin').size().reset_index(name='count')
binned_price_df['price_bin'] = binned_price_df['price_bin'].astype(str)

# Streamlit의 내장 막대 차트(st.bar_chart)를 사용하여 히스토그램 효과를 냅니다.
st.bar_chart(binned_price_df, x="price_bin", y="count", use_container_width=True)

st.markdown("---")


# --- 간단한 통계 계산 ---
st.header("통계 요약")
stats_query = """
    SELECT
        COUNT(*) AS total_books,
        MIN(price) AS min_price,
        MAX(price) AS max_price,
        AVG(price) AS avg_price
    FROM book_metadata;
"""
stats_df = con.execute(stats_query).fetchdf()
st.dataframe(stats_df)