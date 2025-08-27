# etl_book_pipeline.py
# -*- coding: utf-8 -*-

import os
import re
import time
import threading
import requests
import duckdb
import pandas as pd
import numpy as np
import torch
from PIL import Image
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# -----------------
# Global Variables & Constants
# -----------------
START_URL = "http://books.toscrape.com/catalogue/page-1.html"
BASE_URL = "http://books.toscrape.com/"
MAX_PAGES = 50 # 약 1000개 데이터 처리 (20개/페이지 * 50페이지)
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_WORKERS = 16
PER_REQUEST_PAUSE = 0.03
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DUCKDB_DB_FILE = "book_data.duckdb"
MILVUS_COLLECTION_NAME = 'book_images_vectors'
BATCH_SIZE = 100

print_lock = threading.Lock()

# -----------------
# Utility Functions for Scraper
# -----------------
def make_session():
    """요청 세션을 생성하고 재시도 정책을 설정합니다."""
    s = requests.Session()
    retries = Retry(
        total=3, connect=3, read=3, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s

def safe_slug(text: str) -> str:
    """파일 이름에 안전한 슬러그를 생성합니다."""
    slug = re.sub(r'[^\w\s-]', '', text or '').strip().replace(' ', '-')
    return slug[:120] if slug else "unknown"

def parse_book_detail(html: str, base_url: str) -> dict | None:
    """책 상세 페이지 HTML을 파싱하여 정보를 추출합니다."""
    soup = BeautifulSoup(html, "html.parser")
    try:
        title = soup.find("div", class_="product_main").find("h1").get_text(strip=True)
        price_str = soup.find("p", class_="price_color").get_text(strip=True)
        price = float(re.search(r"[\d.]+", price_str).group())
        img_tag = soup.select_one(".item.active img") or soup.find("img")
        img_rel = img_tag["src"] if img_tag else ""
        image_url = urljoin(base_url, img_rel)
        upc_row = soup.find("th", string="UPC")
        upc = upc_row.find_next_sibling("td").get_text(strip=True) if upc_row else None
        
        return {
            "title": title,
            "upc": upc,
            "image_url": image_url,
            "url": base_url,
            "price": price,
        }
    except Exception as e:
        with print_lock:
            print(f" HTML 파싱 중 오류 발생: {e}")
        return None

def download_image(sess, book_data: dict) -> dict:
    """지정된 URL에서 이미지를 다운로드하여 로컬에 저장합니다."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    slug = safe_slug(book_data["title"])
    filename = f"{(book_data['upc'] or slug)}.jpg"
    save_path = os.path.join(IMAGES_DIR, filename)

    try:
        with sess.get(book_data["image_url"], stream=True, timeout=20) as response:
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        book_data["image_path"] = save_path
        return book_data
    except requests.exceptions.RequestException as e:
        with print_lock:
            print(f" 이미지 다운로드 실패 ({book_data['image_url']}): {e}")
        book_data["image_path"] = None
        return book_data

def collect_book_data_from_url(url: str) -> dict | None:
    """단일 URL에서 책 정보와 이미지 다운로드를 처리합니다."""
    sess = make_session()
    try:
        resp = sess.get(url, timeout=15)
        resp.raise_for_status()
        book_data = parse_book_detail(resp.text, url)
        if book_data:
            book_data = download_image(sess, book_data)
        time.sleep(PER_REQUEST_PAUSE)
        return book_data
    except Exception as e:
        with print_lock:
            print(f"(건너김) {url} -> {e}")
        return None
    finally:
        sess.close()

# -----------------
# DuckDB Functions
# -----------------
def setup_duckdb():
    """DuckDB 데이터베이스 및 테이블을 설정합니다."""
    con = duckdb.connect(database=DUCKDB_DB_FILE, read_only=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS book_metadata (
            id VARCHAR PRIMARY KEY,
            title VARCHAR,
            upc VARCHAR,
            price FLOAT,
            url VARCHAR,
            image_path VARCHAR
        );
    """)
    print(f" DuckDB 데이터베이스 '{DUCKDB_DB_FILE}' 및 테이블 설정 완료.")
    return con

def insert_duckdb_batch(con, df: pd.DataFrame):
    """DuckDB에 Pandas DataFrame을 일괄 삽입합니다."""
    if df.empty:
        return
    try:
        con.execute("INSERT INTO book_metadata SELECT * FROM df")
        with print_lock:
            print(f" {len(df)}개 데이터 DuckDB에 배치 삽입 완료.")
    except Exception as e:
        with print_lock:
            print(f" DuckDB에 데이터 삽입 실패: {e}")

# -----------------
# Milvus Functions & Image Encoder
# -----------------
class ImageEncoder:
    """이미지 인코딩 및 폴백 로직을 캡슐화한 클래스."""
    def __init__(self, st_model_name: str = 'clip-ViT-B-32-multilingual-v1'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._st_model = None
        self._supports_images_kw = False
        self._hf_model = None
        self._hf_processor = None
        
        try:
            self._st_model = SentenceTransformer(st_model_name, device=self.device)
            try:
                _dummy = Image.new('RGB', (32, 32))
                _ = self._st_model.encode(images=[_dummy])
                self._supports_images_kw = True
            except (TypeError, Exception):
                self._supports_images_kw = False
        except Exception:
            pass

        if not self._supports_images_kw:
            try:
                self._hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self._hf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                print(" Transformers CLIP 폴백 경로 사용")
            except Exception as e:
                print(f" CLIP 폴백 로드 실패: {e}")
                raise

    def dim(self) -> int:
        return self._st_model.get_sentence_embedding_dimension() if self._supports_images_kw else 512

    @torch.no_grad()
    def encode(self, pil_images: list[Image.Image]) -> np.ndarray:
        if not pil_images:
            return np.zeros((0, self.dim()), dtype=np.float32)

        if self._supports_images_kw:
            embeddings = self._st_model.encode(images=pil_images, normalize_embeddings=True, show_progress_bar=False)
            return embeddings.astype(np.float32)
        else:
            inputs = self._hf_processor(images=pil_images, return_tensors="pt").to(self.device)
            outputs = self._hf_model.get_image_features(**inputs)
            outputs = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
            return outputs.cpu().numpy().astype(np.float32)

def setup_milvus_collection(model_dim: int) -> Collection | None:
    """Milvus 컬렉션을 설정합니다."""
    try:
        connections.connect("default", host="localhost", port="19530")
        if utility.has_collection(MILVUS_COLLECTION_NAME):
            utility.drop_collection(MILVUS_COLLECTION_NAME)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=model_dim),
        ]
        schema = CollectionSchema(fields, "책 이미지 벡터와 UPC")
        collection = Collection(MILVUS_COLLECTION_NAME, schema)
        
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"✨ Milvus 컬렉션 '{MILVUS_COLLECTION_NAME}' 및 인덱스 설정 완료.")
        return collection
    except Exception as e:
        print(f" Milvus 연결 또는 컬렉션 설정 중 오류 발생: {e}")
        return None

def vectorize_and_insert_milvus_batch(milvus_collection, encoder, book_data_list):
    """이미지를 벡터화하고 Milvus에 배치 삽입합니다."""
    if not book_data_list:
        return
        
    images_to_encode = []
    milvus_ids = []
    
    for book in book_data_list:
        p = book.get("image_path")
        upc = book.get("upc")
        # UPC가 있고, 이미지가 실제로 존재하는 경우에만 처리
        if p and upc and os.path.exists(p):
            try:
                image = Image.open(p).convert("RGB")
                images_to_encode.append(image)
                milvus_ids.append(upc)
            except Exception as e:
                with print_lock:
                    print(f" '{book.get('title', 'N/A')}' 이미지 로드 실패: {e}")

    if not images_to_encode:
        return
    
    try:
        embeddings = encoder.encode(images_to_encode)
    except Exception as e:
        with print_lock:
            print(f" 이미지 벡터화 실패: {e}")
            return
    
    try:
        milvus_collection.insert([
            milvus_ids,
            embeddings.tolist()
        ])
        with print_lock:
            print(f" {len(milvus_ids)}개 데이터 Milvus에 배치 삽입 완료.")
    except Exception as e:
        with print_lock:
            print(f" Milvus에 데이터 삽입 실패: {e}")

# -----------------
# Main Logic
# -----------------
if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # 데이터베이스 초기화 및 설정
    duckdb_con = setup_duckdb()
    try:
        encoder = ImageEncoder('clip-ViT-B-32-multilingual-v1')
        print(" 이미지 임베딩 엔진 준비 완료.")
    except Exception as e:
        print(f" 이미지 임베딩 엔진 준비 실패: {e}")
        raise SystemExit(1)
        
    model_dim = encoder.dim()
    milvus_collection = setup_milvus_collection(model_dim)
    if milvus_collection is None:
        raise SystemExit(1)

    print(" 전체 ETL 프로세스 시작...")

    # 1) 전체 URL 수집
    book_urls = []
    page_url = START_URL
    page_count = 0
    sess = make_session()
    while page_url and page_count < MAX_PAGES:
        try:
            resp = sess.get(page_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for article in soup.find_all("article", class_="product_pod"):
                a = article.find("h3").find("a")
                book_urls.append(urljoin(page_url, a["href"]))
            
            next_li = soup.find("li", class_="next")
            page_url = urljoin(page_url, next_li.find("a")["href"]) if next_li else None
            page_count += 1
            print(f"[PAGE {page_count}] {len(book_urls)} URLs collected so far.")
            time.sleep(PER_REQUEST_PAUSE)
        except requests.exceptions.RequestException as e:
            print(f"(페이지 건너김) {page_url} -> {e}")
            break
    sess.close()
    
    print(f"[INFO] 총 수집된 책 URL 수: {len(book_urls)}")
    
    # 2) 멀티스레드 상세 파싱 및 데이터 적재
    processed_books = []
    total_urls = len(book_urls)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(collect_book_data_from_url, url): url for url in book_urls}
        
        for i, future in enumerate(as_completed(futures)):
            book_data = future.result()
            if book_data:
                processed_books.append(book_data)
            
            # 100개 단위로 배치 처리
            if len(processed_books) >= BATCH_SIZE or (i + 1) == total_urls:
                if processed_books:
                    # DuckDB에 메타데이터 삽입
                    df_duck = pd.DataFrame([
                        {
                            "id": b.get("upc") or safe_slug(b.get("title")),
                            "title": b.get("title"),
                            "upc": b.get("upc"),
                            "price": b.get("price"),
                            "url": b.get("url"),
                            "image_path": b.get("image_path")
                        } for b in processed_books
                    ])
                    insert_duckdb_batch(duckdb_con, df_duck)
                    
                    # Milvus에 벡터 데이터 삽입
                    vectorize_and_insert_milvus_batch(milvus_collection, encoder, processed_books)
                    
                    processed_books.clear()
            
            with print_lock:
                print(f"[PROGRESS] {i+1}/{total_urls} 처리 완료.")

    # 3) 최종 동기화 및 정리
    try:
        milvus_collection.flush()
        milvus_collection.load()
        milvus_count = milvus_collection.num_entities
        duckdb_count = duckdb_con.execute("SELECT COUNT(*) FROM book_metadata").fetchone()[0]
        
        duckdb_con.close()
        
        print(f" 최종 결과: Milvus에 {milvus_count}개, DuckDB에 {duckdb_count}개 데이터 적재 완료.")
    except Exception as e:
        print(f" 최종 동기화 및 카운트 중 오류 발생: {e}")

    print("ETL 프로세스 종료.")