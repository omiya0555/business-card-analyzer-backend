import os
import dotenv
import base64
import json
from openai import OpenAI

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url=os.getenv("UPSTAGE_INFO_EXTRACT_URL")
)

def extract_information(file_bytes, mime_type="image/png"):
    """
    Upstage Universal Information Extraction APIを使って
    名刺画像から name, company_name, position を抽出します。
    """
    base64_data = base64.b64encode(file_bytes).decode("utf-8")

    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The full name on the business card"
            },
            "company_name": {
                "type": "string",
                "description": "The company name on the business card"
            },
            "position": {
                "type": "string",
                "description": "The job title or position on the business card"
            }
        }
    }

    try:
        extraction_response = client.chat.completions.create(
            model="information-extract",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_data}"}
                        }
                    ]
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "business_card_schema",
                    "schema": schema
                }
            }
        )

        content = extraction_response.choices[0].message.content
        result = json.loads(content)
        return result

    except Exception as e:
        return {"error": f"Failed to extract information: {str(e)}"}
