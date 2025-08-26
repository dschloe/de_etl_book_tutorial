import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
from sentence_transformers import SentenceTransformer

def process_image_with_milvus():
    """
    이미지를 다운로드하고 벡터화하여 Milvus에 저장한 후, 다시 불러와서 시각화하는 함수
    """
    
    # 1. 이미지 다운로드 및 로컬에 저장
    image_url = 'http://books.toscrape.com/media/cache/fe/72/fe72f0532301ec28892ae79a629a293c.jpg'
    local_image_path = 'downloaded_image.jpg'
    
    print("이미지를 다운로드 중입니다...")
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        with open(local_image_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"이미지가 성공적으로 다운로드되어 '{local_image_path}'에 저장되었습니다.")
    except requests.exceptions.RequestException as e:
        print(f"이미지 다운로드 중 오류가 발생했습니다: {e}")
        return

    # 2. 이미지 벡터화 (Sentence-Transformers 사용)
    print("이미지를 벡터화하는 중입니다...")
    try:
        model = SentenceTransformer('clip-ViT-B-32')

        image = Image.open(local_image_path).convert("RGB")
        vector = model.encode(image)
        
        print(f"이미지 벡터화가 완료되었습니다. 벡터 크기: {len(vector)}")

    except Exception as e:
        print(f"이미지 벡터화 중 오류가 발생했습니다: {e}")
        return

    # 3. Milvus에 벡터 저장 및 인덱스 생성
    print("Milvus에 연결하고 벡터를 저장하는 중입니다...")
    collection_name = 'image_vectors_easier'
    vector_dim = len(vector)
    
    try:
        connections.connect("default", host="localhost", port="19530")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=256)
        ]
        schema = CollectionSchema(fields, "이미지 벡터를 저장하는 컬렉션")
        
        if utility.has_collection(collection_name):
            # 수정된 부분: connections.get_collection() -> Collection(name)
            Collection(collection_name).drop()
        collection = Collection(collection_name, schema)

        data = [[vector], [local_image_path]]
        collection.insert(data)
        collection.flush()
        print(f"벡터와 이미지 경로가 Milvus 컬렉션 '{collection_name}'에 성공적으로 저장되었습니다.")

        index_params = {
            "metric_type": "IP", 
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        print("인덱스 생성이 완료되었습니다.")
    
    except Exception as e:
        print(f"Milvus 작업 중 오류가 발생했습니다. Milvus 서버가 실행 중인지 확인하세요: {e}")
        return
        
    # 4. Milvus에서 벡터 및 관련 메타데이터 불러오기
    print("Milvus에서 저장된 벡터를 다시 불러오는 중입니다...")
    try:
        # 수정된 부분: Collection(name)
        collection = Collection(collection_name)
        collection.load()

        results = collection.query(
            expr="id >= 0",
            output_fields=["image_path"],
            consistency_level="Strong"
        )
        
        retrieved_path = results[0]["image_path"]
        print(f"Milvus에서 불러온 이미지 경로: {retrieved_path}")

        retrieved_image = Image.open(retrieved_path)
        print("Milvus를 통해 가져온 경로의 이미지를 성공적으로 로드했습니다.")
        
    except Exception as e:
        print(f"Milvus에서 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return
    
    # 5. Visualize Both Images
    print("Visualizing both images...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        original_image = Image.open(local_image_path)
        axes[0].imshow(original_image)
        axes[0].set_title("1. Image Downloaded Locally")
        axes[0].axis('off')

        axes[1].imshow(retrieved_image)
        axes[1].set_title("2. Image Loaded via Path from Milvus")
        axes[1].axis('off')

        plt.suptitle("Image Comparison: Local vs. Milvus-retrieved Path", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        os.remove(local_image_path)
        print(f"Local file '{local_image_path}' has been deleted.")

    except Exception as e:
        print(f"An error occurred while visualizing the images: {e}")
        return

if __name__ == "__main__":
    process_image_with_milvus()