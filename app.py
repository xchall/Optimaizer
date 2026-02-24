from fastapi import FastAPI, HTTPException, Response, Depends, Request, status, Body
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional, Any, Dict
from itsdangerous import URLSafeSerializer, BadSignature
import mysql.connector
from mysql.connector import Error
import dotenv
from dotenv import load_dotenv
import os
import uvicorn
import requests

from datetime import datetime
from  polza_ai_module import run_with_tools_polza
load_dotenv()

API_KEY = os.getenv("OPTIMIZER_API_KEY")

api_key_header = APIKeyHeader(name="X-API-Key")

async def check_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return api_key

mysql_log=os.getenv("MYSQL_LOG")
mysql_pass=os.getenv("MYSQL_PASS")
mysql_db=os.getenv("MYSQL_DB")

nexara_url = "https://api.nexara.ru/api/v1/audio/transcriptions"
nexara_api_key = os.getenv("NEXARA_API_KEY")
nexara_headers = {
    "Authorization": f"Bearer {nexara_api_key}",
}

DB_CONFIG = {
    "host": "localhost",
    "user": mysql_log,
    "password": mysql_pass,
    "database": mysql_db
}

app = FastAPI()

class PromptIn(BaseModel):
    system_prompt: str

def segments_to_text(segments: list[dict]) -> str:
    # speaker_0: ...\n speaker_1: ...
    return "\n".join(
        f"{s.get('speaker','')}: {s.get('text','')}".strip(": ")
        for s in segments
        if s.get("text")
    )

def transcribe(external_audio_path: str) -> str:
    data = {
        "url": external_audio_path,
        "response_format": "json",
        # "task": "transcribe",
        "task": "diarize",
        "num_speakers": 2,
        "diarization_setting": "telephonic"
    }

    try:
        response = requests.post(nexara_url, headers=nexara_headers, data=data, timeout=90)
        response.raise_for_status()  # выбросит ошибку, если статус не 2xx

        result = response.json()
        text = result.get("text")
        segments = result.get("segments")
        if not text:
            print("⚠️ Nexara вернула ответ без поля 'text':", result)
            return None
        if not segments:
            print("⚠️ Nexara вернула ответ без поля 'segments':", result)
            return text

        return segments_to_text(segments)
    except requests.exceptions.Timeout: # если ждем ответ дольше 90 секунд
        print("⛔ Ошибка: Nexara не ответила вовремя (timeout)")
        return None

    except requests.exceptions.RequestException as e:
        print("⛔ Ошибка HTTP при обращении к Nexara:", str(e))
        return None

    except ValueError:
        print("⛔ Ошибка: Nexara вернула не‑JSON ответ")
        return None

    except Exception as e:
        print("⛔ Непредвиденная ошибка транскрибации:", str(e))
        return None


def db_get_last_time_by_deal_id(deal_id: int) -> int:
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COALESCE(MAX(created_at), 0) AS last_time "
            "FROM `context` WHERE deal_id = %s",
            (deal_id,)
        )
        (last_time,) = cursor.fetchone()
        return int(last_time)

    except Error as e:
        # Ошибка базы
        raise
    finally:
        try:
            if conn is not None and conn.is_connected():
                conn.close()
        except Exception:
            pass

#Сохранение note
def db_insert_context(cursor, deal_id: int, created_at: int, updated_at: int, note_type: str, payload: str | None):
    cursor.execute(
        """
        INSERT INTO `context` (deal_id, created_at, updated_at, note_type, payload)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (deal_id, created_at, updated_at, note_type, payload),
    )

def db_select_context_lt(cursor, deal_id: int, created_at_limit: int) -> list[tuple[Any, ...]]:
    """
    Вернуть все записи из context по deal_id, у которых created_at < created_at_limit.
    Работает внутри текущей транзакции (использует переданный cursor).
    """
    cursor.execute(
        """
        SELECT id, deal_id, created_at, updated_at, note_type, payload
        FROM `context`
        WHERE deal_id = %s AND created_at < %s
        ORDER BY created_at ASC
        """,
        (deal_id, created_at_limit),
    )
    return cursor.fetchall()


def db_select_context_gt(cursor, deal_id: int, created_at_limit: int) -> list[tuple[Any, ...]]:
    """
    Вернуть все записи из context по deal_id, у которых created_at > created_at_limit.
    Работает внутри текущей транзакции (использует переданный cursor).
    """
    cursor.execute(
        """
        SELECT id, deal_id, created_at, updated_at, note_type, payload
        FROM `context`
        WHERE deal_id = %s AND created_at > %s
        ORDER BY created_at ASC
        """,
        (deal_id, created_at_limit),
    )
    return cursor.fetchall()

def db_insert_result(cursor, deal_id: int, created_at: int, updated_at: int, type: str, llm_result: Optional[str], ) -> None:
    cursor.execute(
        """
        INSERT INTO `results` (deal_id, created_at, updated_at, type, llm_result)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (deal_id, created_at, updated_at, type, llm_result),
    )


def db_select_last_result_by_deal_id(cursor, deal_id: int) -> Optional[tuple[int, int, Optional[str]]]:
    """
    Найти самый последний результат по deal_id (с максимальным created_at).
    """
    cursor.execute(
        """
        SELECT deal_id, created_at, llm_result
        FROM `results`
        WHERE deal_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (deal_id,),
    )
    return cursor.fetchone()

def db_select_last_prompt(cursor) -> Optional[tuple[int, Optional[str]]]:
    """
    Возвращает последнюю добавленную запись из prompt: (id, system_prompt) или None.
    """
    cursor.execute(
        """
        SELECT id, system_prompt
        FROM `prompt`
        ORDER BY id DESC
        LIMIT 1
        """
    )
    return cursor.fetchone()

def db_insert_prompt(cursor, system_prompt: str) -> int:
    """
    Вставляет system_prompt в prompt и возвращает id новой записи.
    """
    cursor.execute(
        """
        INSERT INTO `prompt` (system_prompt)
        VALUES (%s)
        """,
        (system_prompt,),
    )
    return int(cursor.lastrowid)

def notes_to_string(notes: list[tuple[Any, ...]]) -> str:
    out_str = ""
    for note in notes:
        note_type ="Тип заметки: "
        if note[4] == "call_out":
            note_type += "звонок исходящий"
        elif note[4] == "call_in":
            note_type += "звонок входящий"
        else:
            note_type += "текст"

        normal_time = datetime.fromtimestamp(note[2])
        out_str += f" Время заметки: {normal_time}. {note_type}. Содержание заметки: {note[5]}."
    return out_str

# -------------------- Роуты --------------------

@app.get("/get_last_time_by_deal_id/{deal_id}")
async def get_last_time_by_deal_id(
    deal_id: int,
    api_key: str = Depends(check_api_key),
):
    try:
        last_time = db_get_last_time_by_deal_id(deal_id)
    except Error:
        raise HTTPException(status_code=500, detail="DB error")

    # если нет записей — вернется 0
    return {"deal_id": deal_id, "last_time": last_time}

@app.get("/llm_answer/{deal_id}")
async def get_llm_answer(
    deal_id: int,
    api_key: str = Depends(check_api_key),
):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        result = db_select_last_result_by_deal_id(cursor, deal_id)

        # если записей нет
        if result is None:
            res = db_select_context_gt(cursor, deal_id,
                                       0)
            deal_context = notes_to_string(res)

            # не ищем предыдущий ответ, его не было

            llm_answer = run_with_tools_polza(deal_context)
            last_note_time = db_get_last_time_by_deal_id(deal_id)
            db_insert_result(cursor, deal_id, last_note_time, last_note_time, "common", llm_answer)
            conn.commit()
            return {
                "deal_id": int(deal_id),
                "created_at": int(last_note_time),
                "llm_answer": llm_answer,
            }
        #если записи есть
        found_deal_id, created_at, llm_answer = result
        return {
            "deal_id": int(found_deal_id),
            "created_at": int(created_at),
            "llm_answer": llm_answer,
        }

    except Error:
        raise HTTPException(status_code=500, detail="DB error")

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

@app.post("/generate_tasks_scores")
async def generate_tasks_scores(
    request: Request,
    body: list = Body(...),
    api_key: str = Depends(check_api_key),
):
    conn = None
    cursor = None

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        conn.autocommit = False  # выключили автокоvмит
        cursor = conn.cursor()

        previous_last_note_time = 0
        last_note_time = 0
        flag_for_previous_last_note_time = 1
        common_deal_id = 0
        for item in body:
            created_at = int(item.get("created_at"))
            updated_at = int(item.get("updated_at"))
            deal_id = int(item.get("entity_id"))
            note_type = item.get("note_type")
            if flag_for_previous_last_note_time == 1:
                previous_last_note_time = db_get_last_time_by_deal_id(deal_id)
                common_deal_id = deal_id
                flag_for_previous_last_note_time = 0
            payload_text = None

            # Звонок
            if note_type == "call_out":
                link = item.get("params", {}).get("link")
                if not link:
                    raise ValueError("call_out without link")

                text = transcribe(link)
                if text is None:
                    # <-- вот тут “провал” => откатим всё
                    raise RuntimeError(f"Transcription failed for link: {link}")

                payload_text = text
                last_note_time = created_at
            elif note_type == "call_in":
                link = item.get("params", {}).get("link")
                if not link:
                    raise ValueError("call_in without link")

                text = transcribe(link)
                if text is None:
                    # <-- вот тут “провал” => откатим всё
                    raise RuntimeError(f"Transcription failed for link: {link}")

                payload_text = text
                last_note_time = created_at
            # Текстовая заметка
            elif note_type == "common":
                text = item.get("params", {}).get("text")
                if not text:
                    raise ValueError("common without text")
                payload_text = text
                last_note_time = created_at

            else:
                # если неизвестный note_type — реши сам:
                # либо игнорировать, либо считать ошибкой и откатывать
                raise ValueError(f"Unknown note_type: {note_type}")

            # <-- Сохраняем запись в БД (в рамках транзакции)
            db_insert_context(
                cursor=cursor,
                deal_id=deal_id,
                created_at=created_at,
                updated_at=updated_at,
                note_type=note_type,
                payload=payload_text,
            )
            # Если дошли сюда — всё ок, фиксируем каждый отдельный item
            # Если хоть 1 с ошибкой, все до него уже будут сохранены в бд, а после него не обработаются
            conn.commit()

        # Извлекаем из бд все записи, которые относятся к этой сделке
        if last_note_time == 0:
            return {"status": "empty_notes"}
        if previous_last_note_time == 0:# не было предыдущего контекста, и соответственно предыдущего результатат llm
            res = db_select_context_lt(cursor,common_deal_id, last_note_time+1) # из-за строгого сравнения, чтобы не потерять +60 сек
            deal_context = notes_to_string(res)

            #не ищем предыдущий ответ, его не было

            llm_answer = run_with_tools_polza(deal_context)

        else:
            previous_context = db_select_context_lt(cursor, common_deal_id, previous_last_note_time+1)
            new_context = db_select_context_gt(cursor, common_deal_id, previous_last_note_time)
            # Создаем единый конеткст
            previous_context_str = notes_to_string(previous_context)
            new_context_str = notes_to_string(new_context)
            deal_context_without_previous_result = (" SYSTEM_INFO: старый контекст (раннее известный)" + previous_context_str +
                            " SYSTEM_INFO: далее идет новый контекст (новые заметки в рамках сделки)"
                           +  new_context_str)

            #находим предыдущий ответ модели
            previous_result = db_select_last_result_by_deal_id(cursor, common_deal_id)
            if previous_result is None:
                deal_context = deal_context_without_previous_result
            else:
                deal_context = deal_context_without_previous_result + (f" SYSTEM_INFO: далее идет предыдущий твой ответ, который был"
                                                                       f"основан только на старом контексте, без последних заметок {previous_result[2]}")
            llm_answer = run_with_tools_polza(deal_context)

        # нужно записать ответ в таблицу results
        db_insert_result(cursor, common_deal_id, last_note_time, last_note_time, "common", llm_answer)
        conn.commit()

        return {"status": "ok", "used_context": deal_context, "response": llm_answer}

    except Exception as e:
        # Любая ошибка => откат всей пачки
        try:
            if conn is not None:
                conn.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=str(e))

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


@app.get("/prompt/latest")
async def get_latest_prompt(api_key: str = Depends(check_api_key)):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        row = db_select_last_prompt(cursor)
        if row is None:
            return {"id": 0, "system_prompt": ""}

        pid, system_prompt = row
        return {"id": int(pid), "system_prompt": system_prompt or ""}

    except Error:
        raise HTTPException(status_code=500, detail="DB error")
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


@app.post("/prompt")
async def create_prompt(body: PromptIn, api_key: str = Depends(check_api_key)):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        new_id = db_insert_prompt(cursor, body.system_prompt)
        conn.commit()

        return {"status": "ok", "id": new_id}

    except Error:
        try:
            if conn is not None:
                conn.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="DB error")
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

@app.get("/health")
async def health_check():
    return {"status": "ok"}



if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
)
