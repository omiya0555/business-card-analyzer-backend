"""S3サービス - 画像のアップロードと一時URL生成"""
import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        """S3サービスの初期化"""
        self.bucket_name = os.getenv("AWS_S3_BUCKET_NAME", "business-card-results")
        self.region = os.getenv("AWS_REGION", "ap-northeast-1")
        
        # S3クライアントの初期化
        try:
            # 本番環境用の設定
            self.s3_client = boto3.client(
                's3',
                region_name=self.region,
                config=Config(
                    signature_version='s3v4',
                    retries={'max_attempts': 3}
                )
            )
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise

    def upload_image(self, image_data: bytes, content_type: str = "image/png") -> Optional[str]:
        """
        画像をS3にアップロードし、一時URLを生成
        
        Args:
            image_data: 画像のバイナリデータ
            content_type: 画像のMIMEタイプ
            
        Returns:
            一時URL（署名付きURL）
        """
        try:
            # ユニークなファイル名を生成
            file_extension = self._get_file_extension(content_type)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            object_key = f"business-card-results/{timestamp}_{unique_id}{file_extension}"
            
            # S3にアップロード
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=image_data,
                ContentType=content_type,
                Metadata={
                    'uploaded_at': datetime.now().isoformat(),
                    'content_type': content_type
                }
            )
            
            logger.info(f"Image uploaded successfully: {object_key}")
            
            # 一時URL（署名付きURL）を生成（24時間有効）
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=86400  # 24時間
            )
            
            return presigned_url
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {str(e)}")
            return None

    def _get_file_extension(self, content_type: str) -> str:
        """MIMEタイプから適切なファイル拡張子を取得"""
        mime_to_ext = {
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/gif': '.gif',
            'image/webp': '.webp'
        }
        return mime_to_ext.get(content_type.lower(), '.png')

    def check_bucket_exists(self) -> bool:
        """バケットの存在確認"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError:
            return False

    def create_bucket_if_not_exists(self):
        """バケットが存在しない場合は作成"""
        try:
            if not self.check_bucket_exists():
                if self.region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                logger.info(f"Bucket created: {self.bucket_name}")
            else:
                logger.info(f"Bucket already exists: {self.bucket_name}")
        except ClientError as e:
            logger.error(f"Failed to create bucket: {str(e)}")
            raise
