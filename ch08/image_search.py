# -*- coding: utf-8 -*-
import streamlit as st
import os
import duckdb
import torch
import numpy as np
from PIL import Image
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# -----------------
# Global Constants
# -----------------
MILVUS_COLLECTION_NAME = 'book_images_vectors'
DUCKDB_DB_FILE = 'book_data.duckdb'
TOP_K = 5 # 가장 유사한 책 5권 검색

# -----------------
# Singleton Class for Model & DB Connections
# -----------------
class Singleton:
    """앱의 성능을 위해 모델과 DB 연결을 캐싱하는 싱글톤 클래스."""
    _instance = None
    _encoder = None
    _milvus_collection = None
    _duckdb_con = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
            cls.initialize()
        return cls._instance

    @staticmethod
    def initialize():
        try:
            # 이미지 인코더 로드
            Singleton._encoder = ImageEncoder()
            st.success("✅ 이미지 인코딩 모델 로드 완료!")

            # Milvus 컬렉션 연결 및 로드
            if not connections.has_connection("default"):
                connections.connect("default", host="localhost", port="19530")
            if not utility.has_collection(MILVUS_COLLECTION_NAME):
                st.error("❌ Milvus 컬렉션을 찾을 수 없습니다. ETL을 먼저 실행하세요.")
                st.stop()
            Singleton._milvus_collection = Collection(MILVUS_COLLECTION_NAME)
            Singleton._milvus_collection.load()
            st.success(f"✅ Milvus 컬렉션 '{MILVUS_COLLECTION_NAME}' 로드 완료!")

            # DuckDB 연결
            Singleton._duckdb_con = duckdb.connect(database=DUCKDB_DB_FILE, read_only=True)
            st.success("✅ DuckDB 연결 완료!")
            
        except Exception as e:
            st.error(f"❌ 초기화 중 오류 발생: {e}")
            st.stop()

    @staticmethod
    def get_encoder():
        return Singleton._encoder

    @staticmethod
    def get_milvus_collection():
        return Singleton._milvus_collection

    @staticmethod
    def get_duckdb_con():
        return Singleton._duckdb_con

# etl_book_pipeline.py의 ImageEncoder 클래스를 그대로 사용
# Streamlit의 @st.cache_resource와 유사한 역할을 함
class ImageEncoder:
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

# -----------------
# Main Streamlit App
# -----------------
st.set_page_config(layout="wide", page_title="도서 이미지 검색기")
st.title("📚 도서 이미지 검색기")
st.markdown("---")

# 싱글톤 객체 초기화 (앱 실행 시 한 번만 수행)
singleton_instance = Singleton()

# 사이드바에 파일 업로더 추가
st.sidebar.header("이미지 업로드")
uploaded_file = st.sidebar.file_uploader(
    "유사한 도서를 찾을 이미지를 업로드하세요.", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # 업로드된 이미지 처리
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.sidebar.image(uploaded_image, caption="업로드된 이미지", use_container_width=True)
        
        # 이미지 벡터화
        encoder = singleton_instance.get_encoder()
        uploaded_vector = encoder.encode([uploaded_image])
        
        # Milvus에서 유사성 검색
        milvus_collection = singleton_instance.get_milvus_collection()
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = milvus_collection.search(
            data=uploaded_vector,
            anns_field="vector",
            param=search_params,
            limit=TOP_K,
            output_fields=["id"] # 검색 결과로 UPC만 가져옴
        )
        
        # 검색 결과 추출
        upc_list = [hit['id'] for hit in results[0]]
        st.success(f"🔍 Milvus에서 {len(upc_list)}개의 유사한 책을 찾았습니다!")
        
        # DuckDB에서 메타데이터 조회
        duckdb_con = singleton_instance.get_duckdb_con()
        if upc_list:
            upc_string = ", ".join([f"'{upc}'" for upc in upc_list])
            query = f"SELECT * FROM book_metadata WHERE upc IN ({upc_string});"
            results_df = duckdb_con.execute(query).fetchdf()

            if not results_df.empty:
                st.subheader("가장 유사한 도서")
                cols = st.columns(len(results_df))
                
                for i, row in results_df.iterrows():
                    with cols[i]:
                        # use_column_width를 use_container_width로 대체
                        st.image(row["image_path"], caption=f"UPC: {row['upc']}", use_container_width=True)
                        st.markdown(f"**{row['title']}**")
                        st.write(f"가격: ${row['price']:.2f}")
            else:
                st.warning("유사한 도서의 메타데이터를 찾을 수 없습니다.")
        else:
            st.warning("유사한 도서를 찾지 못했습니다. 다른 이미지를 시도해보세요.")
            
    except Exception as e:
        st.error(f"❌ 검색 중 오류가 발생했습니다: {e}")