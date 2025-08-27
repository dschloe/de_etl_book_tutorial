# -*- coding: utf-8 -*-
"""
Books to Scrape 크롤러 (멀티스레딩 / 5페이지 / 진행상황+ETA / 이미지 다운로드)
- 상세 파싱과 동시에 표지 이미지를 data/images/ 에 저장
- 다운로드된 이미지의 임베딩을 생성하여 Milvus에 저장
"""

import os
import re
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from PIL import Image

START_URL = "http://books.toscrape.com/catalogue/page-1.html"
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_WORKERS = 16
PER_REQUEST_PAUSE = 0.03

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"

print_lock = threading.Lock()

# --- Milvus 및 모델 설정 ---
MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = "book_images_collection"

# 이미지 임베딩 모델 로드
try:
    # Milvus 문서에서 권장하는 모델 중 하나인 'clip-ViT-B-32'를 사용합니다.
    # 이 모델은 이미지와 텍스트를 같은 벡터 공간에 임베딩할 수 있습니다.
    # 한글 처리를 위해 'multilingual-v1' 버전으로 변경합니다.
    image_model = SentenceTransformer("clip-ViT-B-32-multilingual-v1")
    print("✅ 이미지 임베딩 모델 로드 완료.")
except Exception as e:
    print(f"❌ 이미지 임베딩 모델 로드 실패: {e}")
    image_model = None

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

def clean_money(s):
    """통화 기호와 불필요한 문자를 제거합니다."""
    return (s or "").replace("Â", "").strip()

def slugify(text, maxlen=80):
    """파일 이름으로 사용할 수 있도록 텍스트를 정리합니다."""
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:maxlen] if text else "untitled"

def guess_ext_from_url(url, default="jpg"):
    """URL에서 이미지 확장자를 추측합니다."""
    path = urlparse(url).path
    if "." in path:
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in {"jpg", "jpeg", "png", "webp"}:
            return "jpg" if ext == "jpeg" else ext
    return default

def download_image(sess, img_url, title, upc):
    """책 이미지를 다운로드하고 로컬 경로를 반환합니다."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ext = guess_ext_from_url(img_url, default="jpg")
    name_part = (upc or slugify(title))
    filename = f"{name_part}.{ext}"
    dest = IMAGES_DIR / filename

    # 파일명 중복 방지
    if dest.exists():
        i = 2
        while True:
            cand = IMAGES_DIR / f"{name_part}-{i}.{ext}"
            if not cand.exists():
                dest = cand
                break
            i += 1

    try:
        with sess.get(img_url, timeout=20, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return str(dest)
    except Exception as e:
        with print_lock:
            print(f"(이미지 실패) {img_url} -> {e}")
        return None

def parse_book_detail(html, base_url):
    """책 상세 페이지 HTML을 파싱하여 정보를 추출합니다."""
    soup = BeautifulSoup(html, "html.parser")

    details = {}
    table = soup.find("table", class_="table table-striped")
    if table:
        for row in table.find_all("tr"):
            th = row.find("th").get_text(strip=True)
            td = row.find("td").get_text(strip=True)
            details[th] = td

    title = soup.find("div", class_="product_main").find("h1").get_text(strip=True)
    desc_anchor = soup.find("div", id="product_description")
    description = desc_anchor.find_next_sibling("p").get_text(strip=True) if desc_anchor else ""

    img_tag = soup.select_one(".item.active img") or soup.find("img")
    img_rel = img_tag["src"] if img_tag else ""
    image_url = urljoin(base_url, img_rel)

    return {
        "title": title,
        "upc": details.get("UPC"),
        "product_type": details.get("Product Type", ""),
        "price_excl_tax": clean_money(details.get("Price (excl. tax)")),
        "price_incl_tax": clean_money(details.get("Price (incl. tax)")),
        "tax": clean_money(details.get("Tax")),
        "availability": details.get("Availability", ""),
        "num_reviews": details.get("Number of reviews"),
        "description": description,
        "image_url": image_url,
        "url": base_url,
    }

def get_book_details(url):
    """책 상세 정보와 이미지를 다운로드합니다."""
    sess = make_session()
    try:
        resp = sess.get(url, timeout=15)
        resp.raise_for_status()
        data = parse_book_detail(resp.text, url)

        img_saved_path = None
        if data.get("image_url"):
            img_saved_path = download_image(
                sess, data["image_url"], data.get("title", ""), data.get("upc")
            )
        data["image_path"] = img_saved_path

        time.sleep(PER_REQUEST_PAUSE)
        return data
    except Exception as e:
        with print_lock:
            print(f"(건너뜀) {url} -> {e}")
        return None
    finally:
        sess.close()

def collect_book_urls(start_url, max_pages=5):
    """책 목록 페이지를 순회하며 상세 페이지 URL을 수집합니다."""
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

            for art in soup.find_all("article", class_="product_pod"):
                a = art.find("h3").find("a")
                book_rel = a["href"]
                book_url = urljoin(page_url, book_rel)
                urls.append(book_url)

            page_count += 1
            next_li = soup.find("li", class_="next")
            page_url = urljoin(page_url, next_li.find("a")["href"]) if (next_li and page_count < max_pages) else None
            time.sleep(PER_REQUEST_PAUSE)
        except Exception as e:
            with print_lock:
                print(f"(페이지 건너뜀) {page_url} -> {e}")
            break

    sess.close()
    return urls

def parse_details_multithread(book_urls, milvus_client, max_workers=MAX_WORKERS):
    """멀티스레드를 사용하여 책 상세 페이지를 파싱하고 Milvus에 데이터를 추가합니다."""
    results = []
    total = len(book_urls)
    done_cnt = 0
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(get_book_details, u): u for u in book_urls}
        for fut in as_completed(fut_map):
            data = fut.result()
            if data:
                if data.get("image_path") and image_model:
                    try:
                        # 이미지 파일 열기
                        image = Image.open(data["image_path"])
                        # 이미지 임베딩 생성 (벡터화)
                        image_embedding = image_model.encode(image)
                        
                        # Milvus에 데이터 삽입
                        milvus_client.insert(
                            collection_name=MILVUS_COLLECTION_NAME,
                            data=[{
                                "vector": image_embedding.tolist(),
                                "title": data.get("title", "N/A"),
                                "url": data.get("url", "N/A"),
                                "image_url": data.get("image_url", "N/A"),
                                "upc": data.get("upc", "N/A"),
                            }]
                        )
                        
                        with print_lock:
                            print(f"✅ Milvus에 '{data['title']}' 이미지 벡터 추가 완료.")

                    except Exception as e:
                        with print_lock:
                            print(f"❌ '{data.get('title', 'N/A')}' 이미지 처리 또는 Milvus 삽입 실패: {e}")

                results.append(data)

            done_cnt += 1
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / max(1, done_cnt)
            eta = avg_time * (total - done_cnt)

            with print_lock:
                print(f"[PROGRESS] {done_cnt}/{total} 완료 "
                      f"({elapsed:.1f}s 경과, ETA {eta:.1f}s)")

    return results

def create_milvus_collection(milvus_client):
    """Milvus 컬렉션을 생성하거나 재생성합니다."""
    if milvus_client.has_collection(collection_name=MILVUS_COLLECTION_NAME):
        milvus_client.drop_collection(collection_name=MILVUS_COLLECTION_NAME)
        print(f"🗑️ 기존 '{MILVUS_COLLECTION_NAME}' 컬렉션 삭제 완료.")

    # `clip-ViT-B-32-multilingual-v1` 모델의 임베딩 차원은 512입니다.
    milvus_client.create_collection(
        collection_name=MILVUS_COLLECTION_NAME,
        dimension=512,
        primary_field_name="pk",
        vector_field_name="vector",
        auto_id=True,
        schema=[
            {"name": "pk", "type": DataType.INT64, "is_primary": True, "auto_id": True},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "dim": 512},
            {"name": "title", "type": DataType.VARCHAR, "max_length": 256},
            {"name": "url", "type": DataType.VARCHAR, "max_length": 512},
            {"name": "image_url", "type": DataType.VARCHAR, "max_length": 512},
            {"name": "upc", "type": DataType.VARCHAR, "max_length": 64},
        ],
        index_params={
            "field_name": "vector",
            "metric_type": "COSINE", # 이미지 유사도 검색에 적합한 '코사인' 유사도 사용
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
    )
    print(f"✨ '{MILVUS_COLLECTION_NAME}' 컬렉션 생성 완료.")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Milvus 클라이언트 초기화
    try:
        milvus_client = MilvusClient(
            uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
            db_name="default",
            timeout=10
        )
        print(f"🔗 Milvus 클라이언트 연결 성공: {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f"❌ Milvus 클라이언트 연결 실패: {e}")
        milvus_client = None

    if milvus_client:
        create_milvus_collection(milvus_client)

        # 1) URL 수집 (5페이지 한정)
        book_urls = collect_book_urls(START_URL, max_pages=5)
        print(f"[INFO] 수집된 책 URL 수: {len(book_urls)}")

        # 2) 멀티스레드 상세 파싱(+이미지 저장 & Milvus 저장)
        books = parse_details_multithread(book_urls, milvus_client)

        milvus_client.close()

    # 3) 파일 저장 (기존과 동일)
    out_path = DATA_DIR / "books_5pages.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {out_path}")
    print(f"Images saved to {IMAGES_DIR}/")