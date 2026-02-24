import pydantic
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv
import os

from openai import OpenAI
import json
import mysql.connector
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
import mysql.connector
from mysql.connector import Error
from typing import List, Union
import re

load_dotenv()  # загружаем переменные среды из .env файла

weaviate_url = os.getenv("WEAVIATE_URL")  # REST endpoint
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

mysql_log=os.getenv("MYSQL_LOG")
mysql_pass=os.getenv("MYSQL_PASS")
mysql_db=os.getenv("MYSQL_DB")

DB_CONFIG = {
    "host": "localhost",
    "user": mysql_log,
    "password": mysql_pass,
    "database": mysql_db
}

API_POLZA_AI = os.getenv("API_POLZA_AI")

def db_select_last_prompt() -> Optional[tuple[int, Optional[str]]]:
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, system_prompt
            FROM `prompt`
            ORDER BY id DESC
            LIMIT 1
            """
        )
        return cursor.fetchone()
    except Error:
        raise Exception("DB error")

    finally:
        try:
            if cursor is not None:
                cursor.close()
        except Exception:
            pass
        try:
            if conn is not None and conn.is_connected():
                conn.close()
        except Exception:
            pass

def books_vector_search(vec_query: str, database_name: str = "SaunaBooksInfo"):
    search_limit = 7
    collection = client.collections.use(database_name)
    response = collection.query.near_text(
        query=vec_query,
        limit=search_limit
    )
    found_chunks = []
    for obj in response.objects:
        content = obj.properties['content']
        found_chunks.append(str(obj.properties['content']))
        # print(str(obj.properties['content']))
    return found_chunks

class BooksVectorSearch(BaseModel):
    """Возвращает релевантные чанки по запросу"""

    vec_query: str = Field(description="Запрос пользователя, по которому будем искать нужную информацию в векторной базе")

    def process(self):
        return {"info": books_vector_search(self.vec_query)}

def pydantic_to_openai_tool(model_cls, name: str, description: str):
    schema = model_cls.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema
        }
    }

tools = [
    pydantic_to_openai_tool(
        BooksVectorSearch,
        "BooksVectorSearch",
        "Возвращает информацию из книг про бпни"
    ),

]

polza = OpenAI(
    base_url="https://api.polza.ai/api/v1",
    api_key=API_POLZA_AI,
)

# with open("system_prompt.txt", "r", encoding="utf-8") as f:
#     SYSTEM_PROMPT = f.read()

LOCAL_TOOLS = {
    "BooksVectorSearch": BooksVectorSearch
}

def run_with_tools_polza(prompt: str) -> str:
    id, sys_prompt = db_select_last_prompt()
    SYSTEM_PROMPT = str(sys_prompt)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    MAX_ROUNDS = 1
    rounds_left = MAX_ROUNDS
    for iteration_round in range(MAX_ROUNDS+1):

        resp = polza.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )

        msg = resp.choices[0].message
        assistant_msg = {
            "role": "assistant",
            "content": msg.content or "",
        }
        # Если модель хочет вызвать tools
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

            messages.append(assistant_msg)
            rounds_left -= 1
            # print(len(msg.tool_calls))
            for i in range (len(msg.tool_calls)):

                tc = msg.tool_calls[i]
                name = tc.function.name

                # парсим аргументы и запускаем локальный Pydantic-класс
                args = json.loads(tc.function.arguments or "{}")
                ToolClass = LOCAL_TOOLS[name]
                obj = ToolClass(**args)
                tool_result = obj.process()
                print(name, args)

                wrapped_content = {
                    "tool_name": name,
                    "tool_args": args,
                    "content": tool_result,
                    "remains_rounds": rounds_left,
                }

                messages.append({
                    "role": "tool",
                    "name": name,
                    "tool_call_id": tc.id,
                    "content": json.dumps(wrapped_content, ensure_ascii=False),
                })

            continue

        # Если tool_calls нет — это финальный ответ модели
        return msg.content

    return "Ошибка: слишком много раундов tool-calls"
