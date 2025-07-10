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

MD_DIR = "topics"

def load_env():
    load_dotenv()
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME", "topic-chunks"),
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

def process_md(file_path: str, file_name: str, client: openai.OpenAI) -> List[Dict]:
    """Markdown ファイルを読み取り、チャンク化してベクトルとメタデータを生成"""
    with open(file_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    plain_text = markdown_to_plain(md_text)
    chunks = chunk_text(plain_text, max_chars=500)  # 500文字でチャンク化
    vectors = []
    for idx, chunk in enumerate(chunks):
        embedding = embed_text(client, chunk)
        vectors.append({
            "id": f"{file_name}-chunk{idx+1}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "source": file_name,
                "chunk_index": idx + 1
            }
        })
    return vectors

# ---------------------------------------------------------------------------- #
# Main upsert routine
# ---------------------------------------------------------------------------- #

def upsert_topics():
    env = load_env()
    client = init_openai(env["openai_api_key"])
    index = init_pinecone(env["pinecone_api_key"], env["pinecone_index_name"])
    all_vectors = []

    for fname in tqdm(os.listdir(MD_DIR), desc="📄 Processing Markdown files"):
        if not fname.lower().endswith(".md"):
            continue
        path = os.path.join(MD_DIR, fname)
        vectors = process_md(path, fname, client)
        all_vectors.extend(vectors)

    print(f"\n✨ Pineconeに {len(all_vectors)} 件のベクトルをアップロード中...")
    index.upsert(all_vectors)
    print("✅ アップロード完了！")

if __name__ == "__main__":
    upsert_topics()
