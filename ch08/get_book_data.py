# -*- coding: utf-8 -*-
import os
import duckdb
from pymilvus import connections, Collection, utility
import matplotlib.pyplot as plt
from PIL import Image

# -----------------
# Global Constants
# -----------------
MILVUS_COLLECTION_NAME = 'book_images_vectors'
DUCKDB_DB_FILE = 'book_data.duckdb'
IMAGES_DIR = 'data/images'

def get_book_data_by_filename(filename: str):
    """
    이미지 파일 이름을 사용하여 해당 책의 메타데이터 및 벡터 정보를 조회합니다.
    
    Args:
        filename (str): 검색할 이미지 파일 이름 (예: '0ab4b35dcffcffd1.jpg').
        
    Returns:
        dict: 조회된 메타데이터 및 벡터 정보.
              데이터가 없으면 None을 반환합니다.
    """
    duckdb_con = None
    milvus_collection = None
    
    try:
        # 1. DuckDB에서 이미지 경로로 메타데이터 조회
        duckdb_con = duckdb.connect(database=DUCKDB_DB_FILE, read_only=True)
        image_path_query = os.path.join(IMAGES_DIR, filename)
        
        # 파일 이름에서 확장자를 제거하여 UPC로 사용
        upc_from_filename = os.path.splitext(filename)[0]

        query = f"SELECT * FROM book_metadata WHERE upc = '{upc_from_filename}';"
        result = duckdb_con.execute(query).fetchone()

        if not result:
            print(f"❌ DuckDB에서 '{filename}'에 대한 메타데이터를 찾을 수 없습니다.")
            return None
            
        columns = [desc[0] for desc in duckdb_con.description]
        book_data = dict(zip(columns, result))
        print(f"✅ DuckDB에서 메타데이터 조회 완료: {book_data.get('title')}")

        # 2. Milvus에서 UPC로 벡터 정보 조회
        connections.connect("default", host="localhost", port="19530")
        collection = Collection(MILVUS_COLLECTION_NAME)
        collection.load()
        
        vector_query = f"id == '{book_data.get('upc')}'"
        milvus_results = collection.query(
            expr=vector_query,
            output_fields=["vector"],
            limit=1
        )
        
        if milvus_results:
            book_data['vector'] = milvus_results[0]['vector']
            print(f"✅ Milvus에서 벡터 정보 조회 완료.")
        else:
            book_data['vector'] = None
            print(f"❌ Milvus에서 UPC '{book_data.get('upc')}'에 대한 벡터를 찾을 수 없습니다.")

        return book_data

    except Exception as e:
        print(f"데이터 조회 중 오류 발생: {e}")
        return None
    finally:
        if duckdb_con:
            duckdb_con.close()
        try:
            connections.disconnect("default")
        except Exception:
            pass

def display_book_data(book_data: dict):
    """
    조회된 책 데이터를 출력하고 이미지를 시각화합니다.
    """
    if not book_data:
        print("조회할 데이터가 없습니다.")
        return

    print("\n--- 조회된 책 정보 ---")
    print(f"제목: {book_data.get('title')}")
    print(f"UPC: {book_data.get('upc')}")
    print(f"가격: ${book_data.get('price')}")
    print(f"이미지 경로: {book_data.get('image_path')}")
    print(f"벡터(일부): {book_data.get('vector', [])[:5]} ...")

    image_path = book_data.get('image_path')
    if image_path and os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title(f"Title: {book_data.get('title')}\nUPC: {book_data.get('upc')}", loc='left', fontsize=10)
            ax.axis('off')
            plt.show()
        except Exception as e:
            print(f"❌ 이미지 시각화 중 오류 발생: {e}")
    else:
        print(f"❌ 이미지 파일이 없거나 경로가 올바르지 않습니다: {image_path}")

# -----------------
# Main 실행 부분 (테스트용)
# -----------------
if __name__ == '__main__':
    # 예시 파일 이름으로 함수 호출
    test_filename = "0b165bd4b9f42fd5.jpg"
    book_info = get_book_data_by_filename(test_filename)
    if book_info:
        display_book_data(book_info)