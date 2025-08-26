# Milvus Standalone 설치 및 시작
- Docker가 준비되었다면, Milvus Standalone을 단일 컨테이너로 실행할 수 있다. 
```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
```
- Milvus Standalone 시작
```bash
bash standalone_embed.sh start
```