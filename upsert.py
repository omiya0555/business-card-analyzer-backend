from __future__ import annotations

import os
import re
import uuid
import logging
from pathlib import Path
from typing import List, Sequence, Dict
from tqdm import tqdm
from dotenv import load_dotenv

import openai
from pinecone.grpc import PineconeGRPC as Pinecone

# ---------------------------------------------------------------------------- #
# Configuration & logging
# ---------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

# ディレクトリ設定
BOOTH_DIR = "booth"
TOPICS_DIR = "topics"

def load_env():
    load_dotenv()
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_booth_index_name": os.getenv("PINECONE_BOOTH_INDEX_NAME", "booth-chunks"),
        "pinecone_topic_index_name": os.getenv("PINECONE_TOPIC_INDEX_NAME", "topic-chunks"),
    }

# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #

def init_openai(api_key: str):
    openai.api_key = api_key
    return openai.OpenAI(api_key=api_key)

def init_pinecone(api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

def markdown_to_plain(md_text: str) -> str:
    """Markdown をプレーンテキストに変換"""
    # Markdown の簡易変換 (HTMLタグを削除)
    return re.sub(r"<[^>]+>", "", md_text).replace("\n", " ").strip()

def embed_text(client: openai.OpenAI, text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

def chunk_text(text: str, max_chars: int) -> List[str]:
    """テキストを指定した文字数でチャンク化"""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def process_md(file_path: str, file_name: str, client: openai.OpenAI, data_type: str) -> List[Dict]:
    """Markdown ファイルを読み取り、チャンク化してベクトルとメタデータを生成"""
    with open(file_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    plain_text = markdown_to_plain(md_text)
    chunks = chunk_text(plain_text, max_chars=500)  # 500文字でチャンク化
    vectors = []
    for idx, chunk in enumerate(chunks):
        embedding = embed_text(client, chunk)
        vectors.append({
            "id": f"{data_type}-{file_name}-chunk{idx+1}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "source": file_name,
                "chunk_index": idx + 1,
                "data_type": data_type
            }
        })
    return vectors

def process_directory(directory: str, data_type: str, client: openai.OpenAI) -> List[Dict]:
    """指定されたディレクトリのマークダウンファイルを処理"""
    if not os.path.exists(directory):
        LOGGER.warning(f"Directory '{directory}' does not exist. Skipping...")
        return []
    
    all_vectors = []
    for fname in tqdm(os.listdir(directory), desc=f"📄 Processing {data_type} files"):
        if not fname.lower().endswith(".md"):
            continue
        path = os.path.join(directory, fname)
        vectors = process_md(path, fname, client, data_type)
        all_vectors.extend(vectors)
    
    return all_vectors

# ---------------------------------------------------------------------------- #
# Main upsert routine
# ---------------------------------------------------------------------------- #

def upsert_topics():
    """両方のインデックスにデータをアップロード"""
    env = load_env()
    client = init_openai(env["openai_api_key"])
    
    # Booth インデックスの処理
    booth_index = init_pinecone(env["pinecone_api_key"], env["pinecone_booth_index_name"])
    booth_vectors = process_directory(BOOTH_DIR, "booth", client)
    
    if booth_vectors:
        print(f"\n✨ Booth インデックスに {len(booth_vectors)} 件のベクトルをアップロード中...")
        booth_index.upsert(booth_vectors)
        print("✅ Booth データのアップロード完了！")
    else:
        print("⚠️  Booth データが見つかりませんでした")
    
    # Topic インデックスの処理
    topic_index = init_pinecone(env["pinecone_api_key"], env["pinecone_topic_index_name"])
    topic_vectors = process_directory(TOPICS_DIR, "tpics", client)
    
    if topic_vectors:
        print(f"\n✨ Topic インデックスに {len(topic_vectors)} 件のベクトルをアップロード中...")
        topic_index.upsert(topic_vectors)
        print("✅ Topic データのアップロード完了！")
    else:
        print("⚠️  Topic データが見つかりませんでした")
    
    print(f"\n🎉 全体の処理が完了しました！")
    print(f"   - Booth vectors: {len(booth_vectors)}")
    print(f"   - Topic vectors: {len(topic_vectors)}")

if __name__ == "__main__":
    upsert_topics()
