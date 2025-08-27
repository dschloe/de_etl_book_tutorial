# image_etl.py
# -*- coding: utf-8 -*-

import os
import re
import time
import threading
import requests
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
from sentence_transformers import SentenceTransformer

# --- 폴백용 (SentenceTransformer가 images 인자를 지원 안 할 때 사용) ---
import torch
from transformers import CLIPProcessor, CLIPModel

# -----------------
# Global Variables & Constants
# -----------------
START_URL = "http://books.toscrape.com/catalogue/page-1.html"
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_WORKERS = 16
PER_REQUEST_PAUSE = 0.03
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MILVUS_COLLECTION_NAME = 'book_images_vectors'
BATCH_SIZE = 100

print_lock = threading.Lock()

# -----------------
# Utility Functions
# -----------------
def make_session():
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
    import re as _re
    slug = _re.sub(r'[^\w\s-]', '', text or '').strip().replace(' ', '-')
    return slug[:120] if slug else "unknown"

def download_image(sess, image_url: str, title: str, upc: str) -> str | None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    slug = safe_slug(title)
    filename = f"{(upc or slug)}.jpg"
    save_path = os.path.join(IMAGES_DIR, filename)

    try:
        with sess.get(image_url, stream=True, timeout=20) as response:
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        return save_path
    except requests.exceptions.RequestException as e:
        with print_lock:
            print(f"❌ 이미지 다운로드 실패 ({image_url}): {e}")
        return None

def parse_book_detail(html: str, base_url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("div", class_="product_main").find("h1").get_text(strip=True)
    img_tag = soup.select_one(".item.active img") or soup.find("img")
    img_rel = img_tag["src"] if img_tag else ""
    image_url = urljoin(base_url, img_rel)
    upc_row = soup.find("th", string="UPC")
    upc = upc_row.find_next_sibling("td").get_text(strip=True) if upc_row else None
    return {"title": title, "upc": upc, "image_url": image_url, "url": base_url}

def collect_book_urls(start_url: str, max_pages: int = 5) -> list[str]:
    urls = []
    page_url = start_url
    page_count = 0
    sess = make_session()
    while page_url and page_count < max_pages:
        with print_lock:
            print(f"[PAGE] {page_url}")
        try:
            resp = sess.get(page_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for article in soup.find_all("article", class_="product_pod"):
                a = article.find("h3").find("a")
                book_rel = a["href"]
                book_url = urljoin(page_url, book_rel)
                urls.append(book_url)
            page_count += 1
            next_li = soup.find("li", class_="next")
            page_url = urljoin(page_url, next_li.find("a")["href"]) if next_li else None
            time.sleep(PER_REQUEST_PAUSE)
        except requests.exceptions.RequestException as e:
            with print_lock:
                print(f"(페이지 건너뜀) {page_url} -> {e}")
            break
    sess.close()
    return urls

def get_book_details(url: str) -> dict | None:
    sess = make_session()
    try:
        resp = sess.get(url, timeout=15)
        resp.raise_for_status()
        data = parse_book_detail(resp.text, url)
        img_saved_path = download_image(sess, data["image_url"], data["title"], data["upc"])
        data["image_path"] = img_saved_path
        time.sleep(PER_REQUEST_PAUSE)
        return data
    except Exception as e:
        with print_lock:
            print(f"(건너뜀) {url} -> {e}")
        return None
    finally:
        sess.close()

# -----------------
# Image Encoder (with fallback)
# -----------------
class ImageEncoder:
    """
    1) SentenceTransformer(images=...) 지원 시 그 경로 사용
    2) 미지원/오류 시 Transformers CLIPModel로 폴백
    모든 경우 float32 + L2 정규화 반환
    """
    def __init__(self, st_model_name: str = 'clip-ViT-B-32-multilingual-v1'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._st_model = None
        self._supports_images_kw = False
        self._hf_model = None
        self._hf_processor = None

        # 1) ST 모델 로드
        try:
            self._st_model = SentenceTransformer(st_model_name, device=self.device)
            # 호환성 체크: 작은 샘플로 images 인자 사용 가능 여부 확인
            try:
                _dummy = Image.new('RGB', (32, 32), color=(128, 128, 128))
                # 일부 구버전은 images kw 미지원 → TypeError 발생
                _ = self._st_model.encode(images=[_dummy], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
                self._supports_images_kw = True
                print("✅ SentenceTransformer 이미지 인코딩 경로 사용")
            except TypeError:
                self._supports_images_kw = False
                print("ℹ️ SentenceTransformer가 images=... 인자를 지원하지 않음 → 폴백 준비")
            except Exception as e:
                print(f"⚠️ SentenceTransformer 이미지 테스트 실패: {e} → 폴백 준비")
                self._supports_images_kw = False
        except Exception as e:
            print(f"❌ SentenceTransformer 로드 실패: {e}")

        # 2) 필요시 폴백 모델 준비
        if not self._supports_images_kw:
            try:
                # 안정적인 공개 가중치: openai/clip-vit-base-patch32 (출력 512차원)
                self._hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self._hf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                print("✅ Transformers CLIP 폴백 경로 사용")
            except Exception as e:
                print(f"❌ CLIP 폴백 로드 실패: {e}")
                raise

    def dim(self) -> int:
        if self._supports_images_kw:
            return self._st_model.get_sentence_embedding_dimension()
        else:
            # openai/clip-vit-base-patch32 의 image_embeds dim = 512
            return 512

    @torch.no_grad()
    def encode(self, pil_images: list[Image.Image]) -> np.ndarray:
        if not pil_images:
            return np.zeros((0, self.dim()), dtype=np.float32)

        if self._supports_images_kw:
            # SentenceTransformer 이미지 경로
            arr = self._st_model.encode(
                images=pil_images,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            ).astype(np.float32)
            return arr
        else:
            # Transformers CLIP 폴백
            # processor가 PIL 이미지를 배치로 받아 tensor 반환
            batch = self._hf_processor(images=pil_images, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self._hf_model.get_image_features(**batch)  # [B, 512]
            # 정규화
            outputs = outputs / (outputs.norm(dim=-1, keepdim=True) + 1e-12)
            return outputs.cpu().numpy().astype(np.float32)

# -----------------
# Milvus Functions
# -----------------
def setup_milvus_collection(model_dim: int) -> Collection | None:
    try:
        connections.connect("default", host="localhost", port="19530")

        if utility.has_collection(MILVUS_COLLECTION_NAME):
            utility.drop_collection(MILVUS_COLLECTION_NAME)
            print(f"🗑️ 기존 '{MILVUS_COLLECTION_NAME}' 컬렉션 삭제 완료.")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=model_dim),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="upc", dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields, "책 이미지 벡터와 메타데이터")
        collection = Collection(MILVUS_COLLECTION_NAME, schema)

        index_params = {
            "metric_type": "IP",           # Inner Product (정규화된 벡터와 궁합)
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"✨ Milvus 컬렉션 '{MILVUS_COLLECTION_NAME}' 및 인덱스 설정 완료.")
        return collection

    except Exception as e:
        print(f"❌ Milvus 연결 또는 컬렉션 설정 중 오류 발생: {e}")
        return None

def vectorize_and_insert(milvus_collection: Collection, encoder: ImageEncoder, book_data_list: list[dict]):
    """이미지를 벡터화하고 Milvus에 배치 삽입"""
    if not book_data_list:
        return

    images_to_encode = []
    metadata_list = []

    for book in book_data_list:
        p = book.get("image_path")
        if p and os.path.exists(p):
            try:
                image = Image.open(p).convert("RGB")
                images_to_encode.append(image)
                metadata_list.append({
                    "title": book.get("title", "N/A"),
                    "image_path": p,
                    "upc": book.get("upc", "N/A"),
                })
            except Exception as e:
                with print_lock:
                    print(f"❌ '{book.get('title', 'N/A')}' 이미지 로드 실패: {e}")

    if not images_to_encode:
        return

    try:
        embeddings = encoder.encode(images_to_encode)   # (N, dim) float32 normalized
    except Exception as e:
        with print_lock:
            print(f"❌ 이미지 벡터화 실패: {e}")
        return

    vectors = [emb.tolist() for emb in embeddings]
    titles = [m["title"] for m in metadata_list]
    image_paths = [m["image_path"] for m in metadata_list]
    upcs = [m["upc"] for m in metadata_list]

    if vectors:
        data = [vectors, titles, image_paths, upcs]
        try:
            milvus_collection.insert(data)
            with print_lock:
                print(f"✅ {len(vectors)}개 데이터 Milvus에 배치 삽입 완료.")
        except Exception as e:
            with print_lock:
                print(f"❌ Milvus에 데이터 삽입 실패: {e}")

# -----------------
# Main Logic
# -----------------
if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # 이미지 인코더(자동 폴백 내장)
    try:
        encoder = ImageEncoder('clip-ViT-B-32-multilingual-v1')
        print("✅ 이미지 임베딩 엔진 준비 완료.")
    except Exception as e:
        print(f"❌ 이미지 임베딩 엔진 준비 실패: {e}")
        raise SystemExit(1)

    model_dim = encoder.dim()
    milvus_collection = setup_milvus_collection(model_dim)
    if milvus_collection is None:
        raise SystemExit(1)

    print("🚀 ETL 프로세스 시작...")

    # 1) URL 수집
    book_urls = collect_book_urls(START_URL, max_pages=5)
    print(f"[INFO] 수집된 책 URL 수: {len(book_urls)}")

    # 2) 멀티스레드 상세 파싱 및 이미지 다운로드 + 배치 벡터화/적재
    processed_books = []
    total_urls = len(book_urls)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_book_details, url): url for url in book_urls}

        for i, future in enumerate(as_completed(futures)):
            book_data = future.result()
            if book_data and book_data.get("image_path"):
                processed_books.append(book_data)

            if len(processed_books) >= BATCH_SIZE:
                print(f"📚 {len(processed_books)}개의 책 데이터를 벡터화 및 Milvus에 적재 중...")
                vectorize_and_insert(milvus_collection, encoder, processed_books)
                processed_books.clear()

            with print_lock:
                print(f"[PROGRESS] {i+1}/{total_urls} 처리 완료.")

        if processed_books:
            print(f"📚 {len(processed_books)}개의(마지막 배치) 책 데이터를 벡터화 및 Milvus에 적재 중...")
            vectorize_and_insert(milvus_collection, encoder, processed_books)
            processed_books.clear()

    # 3) 동기화/로딩/카운트
    try:
        milvus_collection.flush()
        milvus_collection.load()
        count = milvus_collection.num_entities
        print(f"🎉 총 {count}개의 데이터가 Milvus에 성공적으로 적재되었다.")
    except Exception as e:
        print(f"❌ 데이터 flush/load 또는 카운트 중 오류 발생: {e}")

    print("ETL 프로세스 종료.")
