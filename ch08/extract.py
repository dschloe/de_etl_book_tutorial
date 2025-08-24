# extract.py
# -*- coding: utf-8 -*-
"""
Books to Scrape 크롤러

기능:
- 멀티스레딩으로 도서 상세 페이지 병렬 수집
- CLI로 페이지 수 지정(-p/--pages), 최소 5페이지 이상만 허용
- 전체 페이지 수보다 크게 입력되면 전체 페이지 수로 자동 조정하고 알림
- 진행상황 표시 (ETA 포함)
- 도서 이미지 다운로드 및 JSON 데이터 저장
"""

import os
import re
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

START_URL = "http://books.toscrape.com/catalogue/page-1.html"
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_WORKERS = 16
PER_REQUEST_PAUSE = 0.03

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"

print_lock = threading.Lock()

def make_session():
    """
    HTTP 요청용 Session 객체를 생성한다.
    - 재시도 정책(3회)과 백오프, 상태코드 재시도를 설정한다.
    - User-Agent 헤더를 포함한다.
    """
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
    """
    가격 문자열의 특수문자(Â 등)를 제거하고 공백을 정리한다.
    """
    return (s or "").replace("Â", "").strip()

def slugify(text, maxlen=80):
    """
    문자열을 안전한 파일명으로 변환한다.
    - 특수문자 제거, 공백/언더스코어 → 하이픈
    - 최대 길이 제한
    """
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:maxlen] if text else "untitled"

def guess_ext_from_url(url, default="jpg"):
    """
    이미지 URL에서 확장자를 추출한다.
    jpg/jpeg/png/webp만 허용, 그 외에는 기본값 반환
    """
    path = urlparse(url).path
    if "." in path:
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in {"jpg", "jpeg", "png", "webp"}:
            return "jpg" if ext == "jpeg" else ext
    return default

def download_image(sess, img_url, title, upc):
    """
    이미지 파일을 다운로드한다.
    - 파일명은 UPC가 있으면 UPC, 없으면 제목 슬러그를 사용
    - 중복 시 -2, -3 형식으로 추가 저장
    - 저장 실패 시 None 반환
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ext = guess_ext_from_url(img_url, default="jpg")
    name_part = (upc or slugify(title))
    dest = IMAGES_DIR / f"{name_part}.{ext}"

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
    """
    도서 상세 페이지 HTML에서 정보를 추출한다.
    - 제목, UPC, 상품 타입, 가격, 세금, 재고, 리뷰 수, 설명, 이미지 URL
    """
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
    """
    도서 상세 페이지 요청 및 파싱
    - HTML 요청 → parse_book_detail로 정보 추출
    - 이미지 다운로드까지 수행 후 image_path 추가
    """
    sess = make_session()
    try:
        resp = sess.get(url, timeout=15)
        resp.raise_for_status()
        data = parse_book_detail(resp.text, url)
        if data.get("image_url"):
            data["image_path"] = download_image(sess, data["image_url"], data.get("title", ""), data.get("upc"))
        time.sleep(PER_REQUEST_PAUSE)
        return data
    except Exception as e:
        with print_lock:
            print(f"(건너뜀) {url} -> {e}")
        return None
    finally:
        sess.close()

def get_total_pages(start_url):
    """
    사이트 전체 페이지 수를 반환한다.
    - 'Page X of Y' 텍스트에서 Y를 추출
    - 실패 시 None 반환
    """
    sess = make_session()
    try:
        resp = sess.get(start_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        current = soup.select_one("li.current")  # 예: "Page 1 of 50"
        if current:
            m = re.search(r"of\s+(\d+)", current.get_text(strip=True), re.I)
            if m:
                return int(m.group(1))
        return None
    except Exception as e:
        with print_lock:
            print(f"(경고) 전체 페이지 수 확인 실패 -> {e}")
        return None
    finally:
        sess.close()

def collect_book_urls(start_url, max_pages):
    """
    지정된 페이지 수(max_pages)만큼 페이지를 순회하며 도서 상세 URL을 수집한다.
    """
    urls = []
    page_url = start_url
    page_count = 0
    sess = make_session()
    while page_url and page_count < max_pages:
        with print_lock:
            print(f"[PAGE] {page_url}")
        resp = sess.get(page_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for art in soup.find_all("article", class_="product_pod"):
            a = art.find("h3").find("a")
            book_rel = a["href"]
            urls.append(urljoin(page_url, book_rel))

        page_count += 1
        next_li = soup.find("li", class_="next")
        page_url = urljoin(page_url, next_li.find("a")["href"]) if (next_li and page_count < max_pages) else None
        time.sleep(PER_REQUEST_PAUSE)
    sess.close()
    return urls

def parse_details_multithread(book_urls, max_workers=MAX_WORKERS):
    """
    멀티스레딩(ThreadPoolExecutor)으로 도서 상세 페이지를 병렬 파싱한다.
    - 진행상황과 ETA를 출력한다.
    """
    results = []
    total = len(book_urls)
    done_cnt = 0
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(get_book_details, u): u for u in book_urls}
        for fut in as_completed(fut_map):
            data = fut.result()
            if data:
                results.append(data)
            done_cnt += 1
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / max(1, done_cnt)
            eta = avg_time * (total - done_cnt)
            with print_lock:
                print(f"[PROGRESS] {done_cnt}/{total} 완료 ({elapsed:.1f}s 경과, ETA {eta:.1f}s)")
    return results

def parse_args():
    """
    CLI 인자 파서
    -p/--pages : 수집할 페이지 수 (기본 5, 최소 5)
    """
    ap = argparse.ArgumentParser(description="Books to Scrape 크롤러 (멀티스레딩/이미지 저장)")
    ap.add_argument("-p", "--pages", type=int, default=5, help="수집할 페이지 수(최소 5). 기본 5")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 전체 페이지 수 확인 및 입력값 정규화
    total_pages = get_total_pages(START_URL)
    req_pages = args.pages if args.pages and args.pages >= 5 else 5
    if args.pages and args.pages < 5:
        print(f"[알림] 최소 5페이지 이상만 가능하므로 5로 설정한다.")
    if total_pages:
        if req_pages > total_pages:
            print(f"[알림] 요청 페이지({req_pages})가 전체 페이지({total_pages})보다 큼. {total_pages}로 조정한다.")
            req_pages = total_pages
        print(f"[INFO] 전체 페이지: {total_pages}, 수집할 페이지: {req_pages}")
    else:
        print(f"[경고] 전체 페이지 수를 확인하지 못했음. 요청값 {req_pages}페이지만 진행한다.")

    # 디렉터리 준비
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # URL 수집
    book_urls = collect_book_urls(START_URL, max_pages=req_pages)
    print(f"[INFO] 수집된 책 URL: {len(book_urls)}건")

    # 상세 파싱(+이미지 저장)
    books = parse_details_multithread(book_urls)

    # 저장
    out_path = DATA_DIR / f"books_{req_pages}pages.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {out_path}")
    print(f"Images saved to {IMAGES_DIR}/")