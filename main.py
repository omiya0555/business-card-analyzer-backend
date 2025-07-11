"""main.py – FastAPI + Strands Agents

* 名刺画像解析 → 人物・会社概要抽出
* `upsert_topics.init_topics()` をアプリ起動時に呼び出して Pinecone 更新
* Strands Tools: Google Custom Search 検索, Pinecone ベクトル検索
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import logging
import uuid
from typing import List

import dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build
from openai import OpenAI
from pinecone import Pinecone
from strands import Agent, tool
from pydantic import BaseModel

from s3_service import S3Service

from info_extract import extract_information
from upsert_topics import upsert_topics

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
dotenv.load_dotenv()

TOP_K           = int(os.getenv("TOP_K", 3))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

GOOGLE_API_KEY          = os.getenv("GOOGLE_API_KEY")
NAME_COMPANY_SEARCH     = os.getenv("NAME_COMPANY_SEARCH")
FUSIC_SOLUTIONS_SEARCH  = os.getenv("FUSIC_SOLUTIONS_SEARCH")
CSE_SEARCH_NUMBER       = int(os.getenv("SEARCH_NUMBER", 3))

ALLOW_ORIGINS = ["http://localhost:5173, https://686f4cdcb5a535d805e636e3--business-card-analyzer.netlify.app"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# External clients
# ---------------------------------------------------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index = pinecone.Index(PINECONE_INDEX_NAME)

# S3サービスの初期化
s3_service = S3Service()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_query(text: str) -> List[float]:
    rsp = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return rsp.data[0].embedding

# ---------------------------------------------------------------------------
# Strands Tools
# ---------------------------------------------------------------------------

@tool
def search_topic_from_index(query: str) -> str:
    """Pinecone ベクトル検索でブース情報とご興味のありそうな情報を取得"""
    vec = embed_query(query)
    rsp = index.query(vector=vec, top_k=TOP_K, include_metadata=True)
    return "\n---\n".join(m.metadata["text"] for m in rsp.matches) if rsp.matches else "関連トピックが見つかりませんでした。"


@tool
def search_name_and_company(name: str, company_name: str) -> str:
    if not name and not company_name:
        return "検索結果がありませんでした。名前と会社名が指定されていません。"
    svc = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    items = (
        svc.cse()
        .list(q=f"{name} {company_name}".strip(), cx=NAME_COMPANY_SEARCH, num=CSE_SEARCH_NUMBER)
        .execute()
        .get("items", [])
    )
    if not items:
        return f"{name} / {company_name} に関連する情報は見つかりませんでした。"
    return "\n".join(f"- {i['title']}: {i['link']}" for i in items)


@tool
def get_fusic_solutions(query: str) -> str:
    if not query:
        return "検索結果がありませんでした。クエリが指定されていません。"
    try:
        svc = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        response = svc.cse().list(q=query, cx=FUSIC_SOLUTIONS_SEARCH, num=CSE_SEARCH_NUMBER).execute()
        items = response.get("items", [])
        if not items:
            return f"クエリ '{query}' に関連するFusicの開発事例は見つかりませんでした。"
        return "\n".join(f"- {i['title']}: {i['link']}" for i in items)
    except Exception as e:
        LOG.error("Google Custom Search API の呼び出し中にエラーが発生しました: %s", e)
        return f"クエリ '{query}' の検索中にエラーが発生しました: {str(e)}"

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = Agent(
    tools=[search_name_and_company, search_topic_from_index, get_fusic_solutions],
    system_prompt="""
あなたは名刺情報を基に以下を実施するアシスタントです。

1. **search_name_and_company** で人物・会社概要を取得
2. **search_topic_from_index** でDiscoveryのイべント情報と最新ITトレンドを元に、ブース情報1つとご興味のありそうな情報を2つ生成
3. **get_fusic_solutions** で Fusic の開発事例を 3 件取得

### 出力フォーマット
    ---
    【人物と会社の要約】
    <人物と会社の要約を記載してください>

    【Discovery Event ＆ その他情報】
    【Discovery Event ＆ その他情報】
    1. ブース情報（太字）
        <ブース情報のまとめを記載してください>
    2. 情報1（太字）
        <情報のまとめを記載してください>
    3. 情報2（太字）
        <情報のまとめを記載してください>

    【Fusicから提案可能な開発事例】
    1. <開発事例1のタイトル（太字）>: <開発事例1のURL>
        情報
    2. <開発事例2のタイトル（太字）>: <開発事例2のURL>
        情報
    3. <開発事例3のタイトル（太字）>: <開発事例3のURL>
        情報
    ---
""",
)

# ---------------------------------------------------------------------------
# Application Initialization
# ---------------------------------------------------------------------------

def initialize_application():
    """アプリケーションの初期化処理"""
    upsert_topics()  # Pinecone のトピックを初期化
    logging.info("Pinecone topics initialized.")

# アプリケーションの初期化を実行
initialize_application()

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    #allow_origins=ALLOW_ORIGINS,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    フロントエンドから受け取った画像を S3 にアップロードし、
    署名付き URL ではなく **“固定の公開 URL”** を返す。
    """
    LOG.info(f"Received file: {file.filename}, Content-Type: {file.content_type}")

    try:
        # 画像データ読み込み
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="画像データが空です")

        # S3 の公開プレフィクスを使い、推測されにくいキー名を付与
        key = f"public/business-card/{uuid.uuid4()}.png"

        # オブジェクトをアップロード（ACL を public-read に）
        s3_service.s3_client.put_object(
            Bucket=s3_service.bucket_name,
            Key=key,
            Body=image_data,
            ContentType=file.content_type or "image/png",
        )

        # 公開 URL を組み立て
        region = s3_service.s3_client.meta.region_name or "us-east-1"
        public_url = (
            f"https://{s3_service.bucket_name}.s3.{region}.amazonaws.com/{key}"
        )

        return {
            "success": True,
            "download_url": public_url,
            "message": "画像が正常にアップロードされました"
        }

    except HTTPException:
        raise
    except Exception as e:
        LOG.error(f"画像アップロードエラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"画像アップロード中にエラーが発生しました: {e}"
        )

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()
    try:
        extracted_info = extract_information(file_bytes=content, mime_type=file.content_type)
        required_fields = ["name", "company_name", "position"]
        if not all(field in extracted_info for field in required_fields):
            return {"error": "名刺情報から以下の項目を取得できませんでした: " + ", ".join(required_fields)}
        prompt = f"""
        以下は名刺から抽出された情報です：
        - 氏名: {extracted_info['name']}
        - 会社名: {extracted_info['company_name']}
        - 役職・部署: {extracted_info['position']}

        この情報をもとに、以下を調査・出力してください。内容は簡潔に要点を押さえて出力してください：

        1. 名刺情報から人物と社概要を取得してください。
        2. Discoveryのイベント情報と最新ITトレンドを元に、ブース情報1つとご興味のありそうな情報を2つ生成
        3. 上記の情報をもとに、Fusic の開発事例から関心がありそうなものを選定・提示してください

        出力形式：
        ---
        【人物と会社の要約】
        <人物と会社の要約を記載してください>

        【Discovery Event ＆ その他情報】
        1. ブース情報
            <ブース情報のまとめを記載してください>
        2. 情報1
            <情報のまとめを記載してください>
        3. 情報2
            <情報のまとめを記載してください>

        【Fusicから提案可能な開発事例】
        1. <開発事例1のタイトル>: <開発事例1のURL>
            情報
        2. <開発事例2のタイトル>: <開発事例2のURL>
            情報
        3. <開発事例3のタイトル>: <開発事例3のURL>
            情報
        ---
        """

        try:
            result = agent(prompt)
            
            if hasattr(result, 'message'):
                final_message_blocks = result.message.get("content", [])
                final_text = "\n".join(block.get("text", "") for block in final_message_blocks if "text" in block)
            else:
                final_text = str(result)
                
            return {"summary": final_text, "extracted_info": extracted_info}
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {"error": f"Failed to process image: {str(e)}", "details": error_details}

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {"error": f"Failed to process image: {str(e)}", "details": error_details}
