from fastapi import FastAPI, HTTPException, Response, Depends, Request, status, Body
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from typing import Optional, Any, Dict, List
from itsdangerous import URLSafeSerializer, BadSignature
import mysql.connector
from mysql.connector import Error
import dotenv
from dotenv import load_dotenv
import os
import uvicorn
import requests
import httpx
import signal
import re

from datetime import datetime
from  polza_ai_module import run_with_tools_polza
load_dotenv()


# pydantic модели -----

class SelfLink(BaseModel):
    model_config = ConfigDict(extra="ignore")
    href: HttpUrl  # валидируем url

class Links(BaseModel):
    model_config = ConfigDict(extra="ignore")
    self: SelfLink

class Note(BaseModel):
    model_config = ConfigDict(extra="ignore")  # игнорировать лишние поля в ответе
    id: int
    entity_id: int
    created_by: int
    updated_by: int
    created_at: int
    updated_at: int
    responsible_user_id: int
    group_id: int
    note_type: str
    params: Dict[str, Any]  # содержимое не валидируем
    account_id: int
    links: Links = Field(alias="_links")  # в JSON поле "_links"

class Lead(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    name: str


class LeadNotesPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lead: Lead

    # может отсутствовать или быть пустым -> не падаем
    notes: List[Note] = Field(default_factory=list)

    # "54538054": [Note, Note] ... -> тоже может отсутствовать
    contact_notes: Dict[str, List[Note]] = Field(default_factory=dict)

# ---------------------


# Настройка логирования ----------------------
from logging_setup import logger
#------------------------------------------------

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

app = FastAPI(
    title="Optimizer2.0 or Shturman API",
    description="Сервис для генерации скорингов и постановки задач по известным notes из сделки",
    version="1.1.0",
)


class PromptIn(BaseModel):
    system_prompt: str

def segments_to_text(segments: list[dict]) -> str:
    # speaker_0: ...\n speaker_1: ...
    return "\n".join(
        f"{s.get('speaker','')}: {s.get('text','')}".strip(": ")
        for s in segments
        if s.get("text")
    )

def transcribe(external_audio_path: str) -> Optional[str]:
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
        #логи дублирующие
        if not text:
            logger.warning("Nexara вернула ответ без поля 'text': %s", result)
            return None
        if not segments:
            logger.warning("⚠Nexara вернула ответ без поля 'segments': %s", result)
            return text

        return segments_to_text(segments)
    except requests.exceptions.Timeout: # если ждем ответ дольше 90 секунд
        logger.error("Ошибка: Nexara не ответила вовремя (timeout)")
        return None

    except requests.exceptions.RequestException as e:
        logger.error("Ошибка HTTP при обращении к Nexara: %s", e)
        return None

    except ValueError:
        logger.error("Ошибка: Nexara вернула не‑JSON ответ ")
        return None

    except Exception as e:
        logger.error("Непредвиденная ошибка транскрибации: %s", e)
        return None


def db_get_last_time_by_deal_id(cursor, deal_id: int) -> int:
    cursor.execute(
        "SELECT COALESCE(MAX(created_at), 0) AS last_time "
        "FROM `context` WHERE deal_id = %s",
        (deal_id,)
    )
    (last_time,) = cursor.fetchone()
    return int(last_time)



#Сохранение note
def db_insert_context(cursor, note_id: int, deal_id: int, created_at: int, updated_at: int, note_type: str, payload: str | None,
                      processed_ok: int,  # 1 = ok, 0 = fail
                      ):
    cursor.execute(
        """
        INSERT INTO `context` (id, deal_id, created_at, updated_at, note_type, payload, processed_ok)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            payload = VALUES(payload),
            processed_ok = VALUES(processed_ok)
        """,
        (note_id, deal_id, created_at, updated_at, note_type, payload, processed_ok),
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

def db_delete_all_context_by_deal_id(cursor, deal_id: int) -> int:
    """
    Удалить все строки из `context` по deal_id.
    Возвращает количество удалённых строк.
    """
    cursor.execute(
        """
        DELETE FROM `context`
        WHERE deal_id = %s
        """,
        (deal_id,),
    )
    return cursor.rowcount

def db_delete_all_results_by_deal_id(cursor, deal_id: int) -> int:
    """
    Удалить все строки из `results` по deal_id.
    Возвращает количество удалённых строк.
    """
    cursor.execute(
        """
        DELETE FROM `results`
        WHERE deal_id = %s
        """,
        (deal_id,),
    )
    return cursor.rowcount

def db_acquire_deal_lock(cursor, deal_id: int) -> bool:
    """
    Пытается захватить блокировку для указанного deal_id сделки
    Возвращает:
        True  — если блокировка успешно установлена,
        False — если запись уже существует (блокировка занята).
    """

    cursor.execute(
        """
        INSERT INTO deal_processing_locks (deal_id, locked_at)
        VALUES (%s, NOW())
        ON DUPLICATE KEY UPDATE locked_at = locked_at
        """,
        (deal_id,),
    )
    # rowcount == 1 - вставка;
    # rowcount == 2 - duplicate
    return cursor.rowcount == 1

def db_release_deal_lock(cursor, deal_id: int) -> int:
    """
    Освобождает блокировку по deal_id.

    """
    cursor.execute(
        """
        DELETE FROM deal_processing_locks
        WHERE deal_id = %s
        """,
        (deal_id,),
    )
    return cursor.rowcount

def db_get_processed_ok(cursor, note_id: int) -> int | None:
    cursor.execute("SELECT processed_ok FROM `context` WHERE id = %s LIMIT 1", (note_id,))
    row = cursor.fetchone()
    return row[0] if row else None

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

def flatten_notes(payload: LeadNotesPayload) -> List[Note]:
    all_notes: List[Note] = []
    all_notes.extend(payload.notes)

    for _, notes_list in payload.contact_notes.items():
        all_notes.extend(notes_list)

    # чтобы "по порядку" было стабильно и одинаково всегда — сортируем по времени
    all_notes.sort(key=lambda n: n.created_at)
    return all_notes

def check_ai_generated(text):
    pattern = r'^AI Generated Answer'
    return bool(re.match(pattern, text))

# -------------------- Роуты --------------------

@app.get("/get_last_time_by_deal_id/{deal_id}")
async def get_last_time_by_deal_id(
    deal_id: int,
    api_key: str = Depends(check_api_key),
):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        last_time = db_get_last_time_by_deal_id(cursor, deal_id)
    except Error as e:
        logger.error("/get_last_time_by_deal_id/{deal_id} Ошибка базы данных %s", e)
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
            last_note_time = db_get_last_time_by_deal_id(cursor, deal_id)
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
        logger.exception("/llm_answer/{deal_id} упал")
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


EXTERNAL_BASE = "http://217.199.253.86:8000/api/leads/all_data"

@app.get("/generate_tasks_scores/{deal_id}")
async def generate_tasks_scores(
    deal_id: int,
    api_key: str = Depends(check_api_key),
):
    conn = None
    cursor = None

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        conn.autocommit = False  # выключили автокоvмит
        cursor = conn.cursor()

        if not db_acquire_deal_lock(cursor, deal_id):
            return {
                "deal_id": deal_id,
                "locked": True,
            }

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{EXTERNAL_BASE}/{deal_id}")
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Upstream error: {r.status_code}")
        payload = LeadNotesPayload.model_validate(r.json())

        common_deal_id = payload.lead.id
        lead_name = payload.lead.name # Пока не используем

        notes_to_process = flatten_notes(payload)



        previous_last_note_time = db_get_last_time_by_deal_id(cursor, common_deal_id) # до вставки и обновления данных узнаем, какой был

        for note in notes_to_process:
            processed_ok = 1 # флаг, показывающий успешность обработки заметки

            note_id = note.id
            processed = db_get_processed_ok(cursor, note_id)
            if processed == 1:
                continue # пропускаем эту заметку, так как она уже в бд со всей информацией

            created_at = note.created_at
            updated_at = note.updated_at
            entity_id = note.entity_id # не используем
            note_type = note.note_type

            payload_text = None

            if note_type == "call_out": # Исходящий
                link = note.params.get("link")
                if note.params.get("call_status") == 4:
                    if not link:
                        logger.error("call_out without link")
                        text = f"не удалось транскрибировать звонок (нет ссылки на звонок) id = {note_id}"
                        processed_ok = 0
                    else:
                        text = transcribe(link)
                        if text is None: # 1 retry для транскрибации
                            logger.warning("Transcription failed, retry once. link=%s note_id=%s", link, note_id)
                            text = transcribe(link)
                else:
                    text = "Не дозвонились до клиента."
                if text is None:
                    logger.error(f"Transcription failed for link: {link}")
                    text = f"не удалось транскрибировать звонок id = {note_id}"
                    processed_ok = 0

                payload_text = text
            elif note_type == "call_in": # Входящий звонок
                link = note.params.get("link")
                if note.params.get("call_status") == 4:
                    if not link:
                        logger.error("call_in without link")
                        text = f"не удалось транскрибировать звонок (нет ссылки на звонок) id = {note_id}"
                        processed_ok = 0
                    else:
                        text = transcribe(link)
                        if text is None:
                            logger.warning("Transcription failed, retry once. link=%s note_id=%s", link, note_id)
                            text = transcribe(link)
                else:
                    text = "Клиент не дозвонился."
                if text is None:
                    logger.error(f"Transcription failed for link: {link}")
                    text = f"не удалось транскрибировать звонок id = {note_id}"
                    processed_ok = 0

                payload_text = text
            elif note_type == "common":  # Текстовая заметка
                text = note.params.get("text")
                if not text:
                    logger.error("common note without text")
                    text = "содержимое заметки отсутствует"
                    processed_ok = 0
                if check_ai_generated(text): # пропускаем предыдущие ответы Оптимайзера сохраненные внутри CRM в сделке как note с типом common
                    continue
                payload_text = text
            else:
                continue # просто игнорируем attachments и другие

            if processed == 0 and processed_ok == 0: # если уже есть в бд заметка без содержимого, мы не обновляем на неё же без содержимого
                continue

            # Сохраняем заметку note в БД
            db_insert_context(
                cursor=cursor,
                note_id=note_id,
                deal_id=common_deal_id,
                created_at=created_at,
                updated_at=updated_at,
                note_type=note_type,
                payload=payload_text,
                processed_ok=processed_ok,
            )
            # Если дошли сюда — всё ок, фиксируем каждый отдельный note
            conn.commit()

        llm_exc = None
        llm_tb = None
        deal_context = ""

        if previous_last_note_time == 0:# не было предыдущего контекста, и соответственно предыдущего результатат llm
            res = db_select_context_gt(cursor,common_deal_id, previous_last_note_time)
            if not res:
                logger.info(f"/generate_tasks_scores No any context for {common_deal_id}")
                return {
                    "status": "ok",
                    "used_context": "Отсутствует",
                    "response": "AI Generated Answer\n" + "Нет контекста -> нет расчета скоров и постановки задач"
                }

            deal_context = notes_to_string(res)
            #не ищем предыдущий ответ, его не было
            try:
                llm_answer = run_with_tools_polza(deal_context)
            except Exception as e:
                llm_answer = ""
                llm_exc = e
                llm_tb = e.__traceback__

        else:
            previous_context = db_select_context_lt(cursor, common_deal_id, previous_last_note_time+1)
            previous_context_str = notes_to_string(previous_context)
            new_context = db_select_context_gt(cursor, common_deal_id, previous_last_note_time)
            if not new_context:
                # Если уже есть старый ответ LLM вернем именно его, иначе сгенерируем новый
                result = db_select_last_result_by_deal_id(cursor, common_deal_id)
                # если записей нет
                if result is None:
                    res = db_select_context_gt(cursor, common_deal_id,
                                               0)
                    deal_context = notes_to_string(res)

                    # не ищем предыдущий ответ, его не было
                    try:
                        llm_answer = run_with_tools_polza(deal_context)
                    except Exception as e:
                        llm_answer = ""
                        llm_exc = e
                        llm_tb = e.__traceback__
                else:
                    found_deal_id, created_at, llm_answer = result
                    if not llm_answer:
                        res = db_select_context_gt(cursor, common_deal_id,
                                                   0)
                        deal_context = notes_to_string(res)

                        # не ищем предыдущий ответ, его не было
                        try:
                            llm_answer = run_with_tools_polza(deal_context)
                        except Exception as e:
                            llm_answer = ""
                            llm_exc = e
                            llm_tb = e.__traceback__
                    else:
                        logger.info(f"/generate_tasks_scores Took old LLM answer for {common_deal_id}")
                        return {
                            "status": "ok",
                            "used_context": previous_context_str,
                            "response": "AI Generated Answer\n" + llm_answer
                        }
            else:
                # Создаем единый конетекст

                new_context_str = notes_to_string(new_context)
                deal_context_without_previous_result = (" SYSTEM_INFO: старый контекст (раннее известный)" + previous_context_str +
                                " SYSTEM_INFO: далее идет новый контекст (новые заметки в рамках сделки)"
                               +  new_context_str)

                #находим предыдущий ответ модели
                previous_result = db_select_last_result_by_deal_id(cursor, common_deal_id)
                if previous_result is None:
                    deal_context = deal_context_without_previous_result
                else:
                    if not previous_result[2]:
                        deal_context = deal_context_without_previous_result
                    else:
                        deal_context = deal_context_without_previous_result + (f" SYSTEM_INFO: далее идет предыдущий твой ответ, который был"
                                                                           f"основан только на старом контексте, без последних заметок {previous_result[2]}")
                try:
                    llm_answer = run_with_tools_polza(deal_context)
                except Exception as e:
                    llm_answer = ""
                    llm_exc = e
                    llm_tb = e.__traceback__

        last_note_time = db_get_last_time_by_deal_id(cursor, common_deal_id)
        # нужно записать ответ в таблицу results
        db_insert_result(cursor, common_deal_id, last_note_time, last_note_time, "common", llm_answer)
        conn.commit()
        #Добавим в базу даже не удавшуюся генерацию с llm_answer =="", и после только поднимем ошибку если llm_answer==""
        if llm_answer == "":
            if llm_exc is not None:
                raise llm_exc.with_traceback(llm_tb)
            raise RuntimeError("LLM returned empty answer without exception")

        logger.info(f"/generate_tasks_scores successfully for {common_deal_id}")
        return {
            "status": "ok",
            "used_context": deal_context,
            "response": "AI Generated Answer\n" + llm_answer
        }

    except Exception as e:
        logger.exception("/generate_tasks_scores упал")  # traceback в лог
        raise HTTPException(status_code=400, detail=str(e))


    finally:
        # Освобождаем лок
        try:
            if cursor is not None:
                db_release_deal_lock(cursor, deal_id) #высвобождаем сделку от обработки
            if conn is not None and conn.is_connected():
                conn.commit()
        except Exception:
            pass

        # Закрываем курсор
        try:
            if cursor is not None:
                cursor.close()
        except Exception:
            pass

        # Закрываем соединение
        try:
            if conn is not None and conn.is_connected():
                conn.commit()
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
        logger.exception("/prompt/latest не сработал")
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
        logger.exception("/prompt не сработал")
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

@app.delete("/delete_context/{deal_id}")
async def delete_context(
        deal_id: int,
        api_key: str = Depends(check_api_key)
):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        count_deleted = db_delete_all_context_by_deal_id(cursor, deal_id)
        conn.commit()
        return {"status": "ok", "count_deleted": count_deleted }

    except Error:
        logger.exception(f"/delete_context/{deal_id}")
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


@app.delete("/delete_results/{deal_id}")
async def delete_result(
        deal_id: int,
        api_key: str = Depends(check_api_key)
):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        count_deleted = db_delete_all_results_by_deal_id(cursor, deal_id)
        conn.commit()
        return {"status": "ok", "count_deleted": count_deleted}

    except Error:
        logger.exception(f"/delete_results/{deal_id}")
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

@app.post("/crash_test_exception")
async def crash_test_exception(
    api_key: str = Depends(check_api_key)
):
    logger.error("Manual crash test exception endpoint called")
    raise RuntimeError("Manual crash test")


@app.post("/crash_test")
async def crash_test(
    api_key: str = Depends(check_api_key)
):
    logger.error("Manual crash test endpoint called")
    os.kill(os.getpid(), signal.SIGKILL)



@app.get("/health")
async def health_check():
    return {"status": "ok"}



if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        access_log=True # логируем дерганья ручек и responses в journal (это через sys.stdout или sys.stderr)
)
