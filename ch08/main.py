# -*- coding: utf-8 -*-
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import duckdb
from pymilvus import connections, Collection, utility
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import pandas as pd
from urllib.parse import quote
import uuid

# -----------------
# 전역 상수 및 설정
# -----------------
MILVUS_COLLECTION_NAME = 'book_images_vectors'
DUCKDB_DB_FILE = 'book_data.duckdb'
TOP_K = 5
TEMPLATES_DIR = "templates"
STATIC_DIR = "data"
SITE_URL = "http://localhost:8000"

app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/data", StaticFiles(directory=STATIC_DIR), name="data")

# 이미지 검색 결과를 저장할 간단한 메모리 캐시
# 실제 서비스에서는 Redis 등을 사용하는 것이 좋습니다.
image_search_cache = {}

# -----------------
# 전역 리소스
# -----------------
GLOBAL_ENCODER = None
GLOBAL_MILVUS_COLLECTION = None
GLOBAL_DUCKDB_CON = None

class ImageEncoder:
    """이미지를 벡터로 변환하는 클래스."""
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
            except Exception as e:
                raise RuntimeError(f"CLIP 폴백 로드 실패: {e}")
    
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
    def dim(self) -> int:
        return self._st_model.get_sentence_embedding_dimension() if self._supports_images_kw else 512

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 무거운 리소스를 한 번만 로드합니다."""
    global GLOBAL_ENCODER, GLOBAL_MILVUS_COLLECTION, GLOBAL_DUCKDB_CON
    try:
        print("⏳ 리소스 초기화 중...")
        GLOBAL_ENCODER = ImageEncoder()
        
        if not connections.has_connection("default"):
            connections.connect("default", host="localhost", port="19530")
            if not utility.has_collection(MILVUS_COLLECTION_NAME):
                print("❌ Milvus 컬렉션을 찾을 수 없습니다. ETL을 먼저 실행하세요.")
                raise RuntimeError("Milvus 컬렉션이 존재하지 않습니다.")
        
        GLOBAL_MILVUS_COLLECTION = Collection(MILVUS_COLLECTION_NAME)
        GLOBAL_MILVUS_COLLECTION.load()
        
        GLOBAL_DUCKDB_CON = duckdb.connect(database=DUCKDB_DB_FILE, read_only=True)
        print("✅ 서버 초기화 완료: 모델 및 DB 연결 성공")
    except Exception as e:
        print(f"❌ 초기화 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 초기화 실패: 데이터베이스 또는 모델 연결 오류")

# -----------------
# API 엔드포인트
# -----------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search/image", response_class=HTMLResponse)
async def search_by_image(request: Request, image: UploadFile = File(...), num_results: int = Form(10), page: int = 1):
    try:
        contents = await image.read()
        uploaded_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        uploaded_vector = GLOBAL_ENCODER.encode([uploaded_image]).tolist()
        
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        # 사용자가 입력한 num_results를 limit으로 사용합니다.
        milvus_results = GLOBAL_MILVUS_COLLECTION.search(
            data=uploaded_vector,
            anns_field="vector",
            param=search_params,
            limit=num_results,
            output_fields=["id"]
        )
        upc_list = [hit['id'] for hit in milvus_results[0]]

        if not upc_list:
            return templates.TemplateResponse("index.html", {"request": request, "results": [], "page": 1, "total_pages": 1})

        session_id = str(uuid.uuid4())
        image_search_cache[session_id] = upc_list
        
        upc_string = ", ".join([f"'{upc}'" for upc in upc_list])
        offset = (page - 1) * TOP_K
        
        results_df = GLOBAL_DUCKDB_CON.execute(
            f"SELECT * FROM book_metadata WHERE upc IN ({upc_string}) LIMIT {TOP_K} OFFSET {offset};"
        ).fetchdf()

        total_count = len(upc_list)
        total_pages = (total_count + TOP_K - 1) // TOP_K
        
        results = results_df.to_dict('records')
        for r in results:
            if r['image_path'].startswith('ch08/data/'):
                clean_path = r['image_path'][len('ch08/'):]
            elif r['image_path'].startswith('data/'):
                 clean_path = r['image_path']
            else:
                 clean_path = f"data/{r['image_path']}"
            r['image_path'] = f"/{clean_path}"
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request, 
                "results": results, 
                "page": page, 
                "total_pages": total_pages,
                "search_type": "image",
                "session_id": session_id
            }
        )
    except Exception as e:
        print(f"이미지 검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/image/paginate", response_class=HTMLResponse)
async def paginate_image_search(request: Request, session_id: str, page: int = 1):
    """캐시된 결과를 기반으로 이미지 검색 페이지네이션을 처리합니다."""
    try:
        upc_list = image_search_cache.get(session_id)
        if not upc_list:
            raise HTTPException(status_code=404, detail="세션이 만료되었거나 찾을 수 없습니다.")

        upc_string = ", ".join([f"'{upc}'" for upc in upc_list])
        offset = (page - 1) * TOP_K

        results_df = GLOBAL_DUCKDB_CON.execute(
            f"SELECT * FROM book_metadata WHERE upc IN ({upc_string}) LIMIT {TOP_K} OFFSET {offset};"
        ).fetchdf()

        total_count = len(upc_list)
        total_pages = (total_count + TOP_K - 1) // TOP_K
        
        results = results_df.to_dict('records')
        for r in results:
            if r['image_path'].startswith('ch08/data/'):
                clean_path = r['image_path'][len('ch08/'):]
            elif r['image_path'].startswith('data/'):
                 clean_path = r['image_path']
            else:
                 clean_path = f"data/{r['image_path']}"
            r['image_path'] = f"/{clean_path}"
            
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": results,
                "page": page,
                "total_pages": total_pages,
                "search_type": "image",
                "session_id": session_id
            }
        )
    except Exception as e:
        print(f"이미지 검색 페이지네이션 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/metadata", response_class=HTMLResponse)
async def search_by_metadata(request: Request, min_price: float = 0, max_price: float = 100, page: int = 1):
    try:
        offset = (page - 1) * TOP_K

        results_df = GLOBAL_DUCKDB_CON.execute(
            f"SELECT * FROM book_metadata WHERE price >= {min_price} AND price <= {max_price} ORDER BY price ASC LIMIT {TOP_K} OFFSET {offset};"
        ).fetchdf()

        total_count = GLOBAL_DUCKDB_CON.execute(
            f"SELECT COUNT(*) FROM book_metadata WHERE price >= {min_price} AND price <= {max_price};"
        ).fetchone()[0]
        total_pages = (total_count + TOP_K - 1) // TOP_K

        results = results_df.to_dict('records')
        for r in results:
            if r['image_path'].startswith('ch08/data/'):
                clean_path = r['image_path'][len('ch08/'):]
            elif r['image_path'].startswith('data/'):
                 clean_path = r['image_path']
            else:
                 clean_path = f"data/{r['image_path']}"
            r['image_path'] = f"/{clean_path}"

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": results,
                "page": page,
                "total_pages": total_pages,
                "search_type": "metadata",
                "min_price": min_price,
                "max_price": max_price
            }
        )
    except Exception as e:
        print(f"메타데이터 검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)