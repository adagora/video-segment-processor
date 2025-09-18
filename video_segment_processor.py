#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import logging
import warnings
import requests
import subprocess
import argparse
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from report_generator import create_comprehensive_video_report

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

RAGFLOW_API_URL = None
RAGFLOW_API_KEY = None


def parse_video_path_with_params(video_path_argument):
    """
    Parsuje ścieżkę wideo z parametrami URL i zwraca ścieżkę oraz opcje
    
    Args:
        video_path_argument: Ścieżka wideo z opcjonalnymi parametrami (np. "video.mp4?timestamps=1")
    
    Returns:
        tuple: (clean_video_path, options_dict)
    """
    if '?' in video_path_argument:
        path_part, query_part = video_path_argument.split('?', 1)
        params = parse_qs(query_part)
        
        timestamps = params.get("timestamps", ["0"])[0] in ("1", "true", "True")
        
        options = {
            "timestamps": timestamps
        }
        
        logger.info(f"Parsowanie ścieżki wideo z parametrami: {path_part}, opcje: {options}")
        return path_part, options
    else:
        return video_path_argument, {"timestamps": False}


def load_environment_config():
    """Załaduj konfigurację środowiska"""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        logger.info(f"Załadowano plik zmiennych środowiskowych: {env_file}")
    else:
        config_file = os.path.join(os.path.dirname(__file__), 'config.env')
        if os.path.exists(config_file):
            load_dotenv(config_file)
            logger.info(f"Załadowano plik konfiguracyjny: {config_file}")
        else:
            logger.warning("Nie znaleziono .env ani config.env, używane będą zmienne systemowe")
    
    global RAGFLOW_API_URL, RAGFLOW_API_KEY
    RAGFLOW_API_URL = os.getenv('RAGFLOW_API_URL', 'http://localhost:9380')
    RAGFLOW_API_KEY = os.getenv('RAGFLOW_API_KEY')
    
    if not RAGFLOW_API_KEY:
        logger.warning("Nie ustawiono RAGFLOW_API_KEY, funkcje RAGFlow będą niedostępne")
    
    logger.info(f"Konfiguracja RAGFlow załadowana: URL={RAGFLOW_API_URL}")


def get_dataset_by_name(name):
    """Pobierz ID zbioru danych RAGFlow na podstawie nazwy"""
    headers = {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }

    response = requests.get(
        f"{RAGFLOW_API_URL}/api/v1/datasets?name={name}",
        headers=headers
    )

    if response.status_code == 200:
        result = response.json()
        if result.get("code") == 0:
            datasets = result.get("data", [])
            for dataset in datasets:
                if dataset.get("name") == name:
                    logger.info(f"Znaleziono istniejący zbiór danych: {name}, ID: {dataset.get('id')}")
                    return dataset.get('id')

    logger.info(f"Nie znaleziono zbioru danych o nazwie {name}")
    return None


GRAPHRAG_CONFIG = {
    "use_graphrag": True,
    "entity_types": ["organization", "person", "geo", "event", "category"],
    "method": "general",
    "community": False,
    "resolution": False
}


def create_ragflow_dataset(name, description="Video segments dataset"):
    """Utwórz zbiór danych RAGFlow z włączonym grafem wiedzy"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }

    payload = {
        "name": name,
        "description": description,
    "chunk_method": "naive",
        "parser_config": {
            "chunk_token_num": 512,
            "delimiter": "\n",
            "filename_embd_weight": 0.1,
            "task_page_size": 5,
            "raptor": {
                "use_raptor": False
            },
            "graphrag": GRAPHRAG_CONFIG
        }
    }

    response = requests.post(
        f"{RAGFLOW_API_URL}/api/v1/datasets",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        if result.get("code") == 0:
            dataset_id = result.get("data", {}).get("id")
            logger.info(f"✅ Pomyślnie utworzono zbiór danych, ID: {dataset_id}")
            logger.info(f"🕸️ Funkcja grafu wiedzy została włączona: {GRAPHRAG_CONFIG}")
            return dataset_id
        else:
            logger.error(f"Nie udało się utworzyć zbioru danych: {result.get('message')}")
            return None
    else:
        logger.error(f"Błąd żądania API, kod statusu: {response.status_code}")
        return None


def verify_dataset_graphrag_config(dataset_id):
    """Sprawdź czy konfiguracja grafu wiedzy w zbiorze danych jest poprawna

    Args:
        dataset_id: ID zbioru danych

    Returns:
        dict: Wynik weryfikacji konfiguracji
    """
    headers = {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }

    try:
        response = requests.get(
            f"{RAGFLOW_API_URL}/api/v1/datasets/{dataset_id}",
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                dataset_data = result.get("data", {})
                parser_config = dataset_data.get("parser_config", {})
                graphrag_config = parser_config.get("graphrag", {})


                chunk_method = dataset_data.get("chunk_method", "unknown")
                parser_id = dataset_data.get("parser_id", "unknown")
                is_graphrag_enabled = graphrag_config.get("use_graphrag", False)

                is_kg_dataset = (chunk_method == "knowledge_graph" and parser_id == "knowledge_graph")

                logger.info(f"📊 Weryfikacja konfiguracji grafu wiedzy dla zbioru {dataset_id}:")
                logger.info(f"   - Metoda chunkowania: {chunk_method}")
                logger.info(f"   - Parser ID: {parser_id}")
                logger.info(f"   - Konfiguracja GraphRAG: {'✅ Włączona' if is_graphrag_enabled else '❌ Wyłączona'}")
                logger.info(f"   - Funkcja grafu wiedzy: {'✅ Włączona' if is_kg_dataset else '❌ Wyłączona'}")

                if is_kg_dataset:
                    method = graphrag_config.get("method", "unknown")
                    entity_types = graphrag_config.get("entity_types", [])
                    logger.info(f"   - Metoda GraphRAG: {method}")
                    logger.info(f"   - Typy encji: {entity_types}")

                return {
                    "success": True,
                    "enabled": is_kg_dataset,
                    "chunk_method": chunk_method,
                    "parser_id": parser_id,
                    "graphrag_config": graphrag_config,
                    "is_graphrag_enabled": is_graphrag_enabled
                }
            else:
                logger.error(f"Nie udało się pobrać szczegółów zbioru danych: {result.get('message')}")
                return {"success": False, "error": result.get('message')}
        else:
            logger.error(f"Błąd zapytania o zbiór danych, kod statusu: {response.status_code}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        logger.error(f"Błąd podczas weryfikacji konfiguracji grafu wiedzy: {e}")
        return {"success": False, "error": str(e)}


def check_document_status(dataset_id, document_id):
    """Sprawdź status przetwarzania dokumentu

    Args:
        dataset_id: ID zbioru danych
        document_id: ID dokumentu

    Returns:
        dict: Informacje o statusie dokumentu {"status": str, "run": str, "progress": float, "progress_msg": str}
    """
    headers = {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }

    try:
        response = requests.get(
            f"{RAGFLOW_API_URL}/api/v1/datasets/{dataset_id}/documents?id={document_id}",
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                docs = result.get("data", {}).get("docs", [])
                if docs:
                    document_info = docs[0]
                    status_info = {
                        "status": document_info.get("status", "0"),
                        "run": document_info.get("run", "0"),
                        "progress": document_info.get("progress", 0.0),
                        "progress_msg": document_info.get("progress_msg", ""),
                        "chunk_count": document_info.get("chunk_count", 0)
                    }
                    return status_info
                else:
                    logger.error(f"Nie znaleziono dokumentu {document_id}")
                    return {"status": "NOT_FOUND", "run": "0", "progress": 0.0, "progress_msg": "", "chunk_count": 0}
            else:
                logger.error(f"Błąd pobierania statusu dokumentu: {result.get('message')}")
                return {"status": "ERROR", "run": "0", "progress": 0.0, "progress_msg": "", "chunk_count": 0}
        else:
            logger.error(f"Błąd pobierania statusu dokumentu, kod statusu: {response.status_code}")
            return {"status": "ERROR", "run": "0", "progress": 0.0, "progress_msg": "", "chunk_count": 0}
    except Exception as e:
        logger.error(f"Błąd podczas pobierania statusu dokumentu: {e}")
        return {"status": "ERROR", "run": "0", "progress": 0.0, "progress_msg": "", "chunk_count": 0}



def wait_for_documents_parsing(dataset_id, document_ids, check_interval=30):
    """Czekaj na zakończenie przetwarzania wszystkich dokumentów (wersja uproszczona, bez pokazywania postępu)

    Args:
        dataset_id: ID zbioru danych
        document_ids: Lista ID dokumentów
        check_interval: Interwał sprawdzania (sekundy), domyślnie 30

    Returns:
        dict: Statystyki przetwarzania
    """
    print(f"\n{'=' * 60}")
    print(f"Oczekiwanie na zakończenie przetwarzania {len(document_ids)} dokumentów...")
    print(f"Interwał sprawdzania: {check_interval} sekund")
    print(f"{'=' * 60}")

    start_time = time.time()
    completed_docs = set()
    failed_docs = set()

    def format_time(seconds):
        """Formatuj czas do wyświetlenia"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        for doc_id in document_ids:
            if doc_id in completed_docs or doc_id in failed_docs:
                continue

            status_info = check_document_status(dataset_id, doc_id)
            chunk_count = status_info.get("chunk_count", 0)

            if chunk_count > 0:
                completed_docs.add(doc_id)
                print(f"✅ Dokument {doc_id} przetworzony (liczba chunków: {chunk_count})")
            elif status_info.get("status") in ["ERROR", "NOT_FOUND"]:
                failed_docs.add(doc_id)
                print(f"❌ Dokument {doc_id} nie został przetworzony")


        remaining = len(document_ids) - len(completed_docs) - len(failed_docs)
        print(f"⏱️  Czas oczekiwania: {format_time(elapsed_time)} | "
              f"✅ Zakończone: {len(completed_docs)} | "
              f"❌ Nieudane: {len(failed_docs)} | "
              f"⏳ Oczekujące: {remaining}")

        if len(completed_docs) + len(failed_docs) == len(document_ids):
            print(f"\n{'=' * 60}")
            print("🎉 Wszystkie dokumenty zostały przetworzone!")
            print(f"{'=' * 60}")
            break

        print(f"Czekam {check_interval} sekund przed kolejnym sprawdzeniem...")
        time.sleep(check_interval)

    result = {
        "total": len(document_ids),
        "completed": len(completed_docs),
        "failed": len(failed_docs),
        "completed_ids": list(completed_docs),
        "failed_ids": list(failed_docs),
        "elapsed_time": time.time() - start_time
    }

    print(f"📊 Statystyki przetwarzania:")
    print(f"   Liczba dokumentów: {result['total']}")
    print(f"   Przetworzone pomyślnie: {result['completed']} ✅")
    print(f"   Nieudane: {result['failed']} ❌")
    print(f"   Całkowity czas: {format_time(result['elapsed_time'])}")

    if result['failed'] > 0:
        print(f"   ID nieudanych dokumentów: {result['failed_ids']}")

    logger.info(f"Oczekiwanie na przetwarzanie dokumentów zakończone - Łącznie: {result['total']}, "
                f"Sukces: {result['completed']}, Nieudane: {result['failed']}, "
                f"Czas: {result['elapsed_time']:.1f}s")

    return result


def parse_documents(dataset_id, document_ids):
    """Przetwarzaj dokumenty w zbiorze danych RAGFlow

    Args:
        dataset_id: ID zbioru danych
        document_ids: Lista ID dokumentów

    Returns:
        bool: Czy udało się wyzwolić przetwarzanie
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }

    payload = {
        "document_ids": document_ids
    }

    try:
        logger.info(f"Rozpoczynam wyzwalanie przetwarzania dokumentów, liczba: {len(document_ids)}")
        logger.info(f"URL żądania: {RAGFLOW_API_URL}/api/v1/datasets/{dataset_id}/chunks")

        response = requests.post(
            f"{RAGFLOW_API_URL}/api/v1/datasets/{dataset_id}/chunks",
            headers=headers,
            json=payload,
            timeout=60
        )

        logger.info(f"Kod statusu odpowiedzi: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pomyślnie wyzwolono przetwarzanie dokumentów, liczba: {len(document_ids)}")
                return True
            else:
                logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Nie udało się wyzwolić przetwarzania dokumentów: {result.get('message')}")
                return False
        else:
            logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Nie udało się wyzwolić przetwarzania dokumentów, kod statusu: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Błąd podczas wyzwalania przetwarzania dokumentów: {e}")
        import traceback
        logger.error(f"Stos błędów: {traceback.format_exc()}")
        return False


def update_document_parser_config(dataset_id, document_id):
    """Aktualizuj konfigurację parsera dokumentu, upewniając się, że zawiera wymagany parser_config

    Args:
        dataset_id: ID zbioru danych
        document_id: ID dokumentu

    Returns:
        bool: Czy aktualizacja się powiodła
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }

    payload = {
    "chunk_method": "naive", 
        "parser_config": {
            "chunk_token_num": 512,
            "delimiter": "\n",
            "filename_embd_weight": 0.1,
            "task_page_size": 5, 
            "graphrag": GRAPHRAG_CONFIG
        }
    }

    try:
        response = requests.put(
            f"{RAGFLOW_API_URL}/api/v1/datasets/{dataset_id}/documents/{document_id}",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                logger.info(f"✅ Pomyślnie zaktualizowano konfigurację parsera dokumentu {document_id}")
                logger.info(f"🕸️ Funkcja GraphRAG została włączona: {GRAPHRAG_CONFIG}")
                return True
            else:
                logger.error(f"Nie udało się zaktualizować konfiguracji parsera dokumentu: {result.get('message')}")
                return True
        else:
            logger.error(f"Nie udało się zaktualizować konfiguracji parsera dokumentu, kod statusu: {response.status_code}")
            return True
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji konfiguracji parsera dokumentu: {e}")
        return True


def upload_file_to_ragflow(dataset_id, file_name, file_path, file_type=None, auto_parse=False):
    """Wyślij plik do zbioru danych RAGFlow

    Args:
        dataset_id: ID zbioru danych
        file_name: Nazwa pliku
        file_path: Ścieżka do pliku
        file_type: Typ pliku, opcjonalnie
        auto_parse: Czy automatycznie przetwarzać, domyślnie False

    Returns:
        str: ID dokumentu, None jeśli niepowodzenie
    """
    headers = {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }

    import mimetypes
    mime_type = mimetypes.guess_type(file_path)[0]
    if mime_type is None:
        if file_path.endswith('.json'):
            mime_type = 'application/json'
        elif file_path.endswith('.mp3'):
            mime_type = 'audio/mpeg'
        elif file_path.endswith('.mp4'):
            mime_type = 'video/mp4'
        elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
            mime_type = 'image/jpeg'
        else:
            mime_type = 'application/octet-stream'

    try:
        with open(file_path, 'rb') as f:
            files = {
                'file': (file_name, f, mime_type)
            }

            params = {}
            if file_type == 'binary':
                params['file_type'] = 'binary'

            logger.info(f"Rozpoczynam wysyłanie pliku {file_name} do zbioru danych {dataset_id}")

            response = requests.post(
                f"{RAGFLOW_API_URL}/api/v1/datasets/{dataset_id}/documents",
                headers=headers,
                files=files,
                params=params,
                timeout=60
            )
    except Exception as e:
        logger.error(f"Nie udało się wysłać pliku: {e}")
        import traceback
        logger.error(f"Stos błędów: {traceback.format_exc()}")
        return None

    if response.status_code == 200:
        result = response.json()
        if result.get("code") == 0:
            document_id = result.get("data", [])[0].get("id") if isinstance(result.get("data"), list) and len(
                result.get("data")) > 0 else None
            logger.info(f"Pomyślnie wysłano plik {file_name}, ID: {document_id}")

            if auto_parse and document_id:
                logger.info(f"Automatyczne wyzwalanie przetwarzania dokumentu {document_id}")
                parse_documents(dataset_id, [document_id])

            return document_id
        else:
            logger.error(f"Nie udało się wysłać pliku {file_name}: {result.get('message')}")
            return None
    else:
        logger.error(f"Nie udało się wysłać pliku {file_name}, kod statusu: {response.status_code}, odpowiedź: {response.text[:200]}...")
        return None


def segment_video(video_path, output_dir, segment_duration=200, timestamps=False):
    """
    Segmentuje wideo i zapisuje do określonego katalogu.

    Args:
        video_path: Ścieżka do pliku wideo.
        output_dir: Katalog wyjściowy.
        segment_duration: Czas trwania każdego segmentu wideo (w sekundach).
        timestamps: Czy dodać znaczniki czasu do wideo.

    Returns:
        Lista słowników z informacjami o segmentach. Każdy słownik zawiera klucze
        takie jak 'mp4_path', 'mp3_path', 'start_time', 'end_time'.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
             video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        duration = float(result.stdout.strip())
        logger.info(f"Całkowity czas trwania wideo: {duration:.2f} sekund")
    except Exception as e:
        logger.error(f"Nie udało się pobrać czasu trwania wideo: {e}")
        return []

    num_segments = int(duration / segment_duration) + (1 if duration % segment_duration > 0 else 0)
    logger.info(f"Podziel wideo na {num_segments} segmentów, każdy po {segment_duration} sekund")

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, duration)

        segment_info = {
            'index': i,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        }

        mp4_path = os.path.join(output_dir, f"{video_name}_segment_{i}.mp4")
        mp3_path = os.path.join(output_dir, f"{video_name}_segment_{i}.mp3")

        try:
            logger.info(f"Wyodrębnianie segmentu wideo {i + 1}/{num_segments}: {start_time:.2f}s - {end_time:.2f}s")
            
            if timestamps:
                base_name = os.path.basename(video_path)
                
                # Format: "video.mp4 [01:00] + PTS" gdzie [01:00] to start_time a PTS to offset
                hours = int(start_time // 3600)
                minutes = int((start_time % 3600) // 60)
                seconds = int(start_time % 60)
                
                if hours > 0:
                    start_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    start_time_str = f"{minutes:02d}:{seconds:02d}"
                
                start_time_str_escaped = start_time_str.replace(":", "\\:")
                text = f"{base_name} {start_time_str_escaped}+%{{pts\\:hms}}"
                draw_filter = (
                    f"drawtext=fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5"
                    f":text='{text}':x=w-tw-10:y=h-th-10"
                )
                
                logger.info(f"Dodawanie znaczników czasu do segmentu wideo (bazowy czas: {start_time_str}) z filtrem drawtext")
                
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-ss', str(start_time),
                    '-to', str(end_time), '-i', video_path,
                    '-vf', draw_filter, '-c:a', 'copy', mp4_path
                ]
                logger.info(f"Komenda ffmpeg: {' '.join(ffmpeg_cmd)}")
                
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                subprocess.run([
                    'ffmpeg', '-y', '-ss', str(start_time),
                    '-to', str(end_time), '-i', video_path,
                    '-c', 'copy', mp4_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            segment_info['mp4_path'] = mp4_path
        except Exception as e:
            logger.error(f"Nie udało się wyodrębnić segmentu wideo: {e}")
            continue

        try:
            logger.info(f"Wyodrębnianie segmentu audio {i + 1}/{num_segments}")
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(start_time),
                '-to', str(end_time), '-i', video_path,
                '-q:a', '0', '-map', 'a', mp3_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            segment_info['mp3_path'] = mp3_path
        except Exception as e:
            logger.error(f"Nie udało się wyodrębnić segmentu audio: {e}")
            continue

        segments.append(segment_info)

    logger.info(f"Pomyślnie wyodrębniono {len(segments)} segmentów wideo")
    return segments


def create_granular_transcript(words_data, segment_start_time=0, timestamp_interval=10, sentence_timestamps=True):
    """
    Tworzy transkrypt z granularnymi znacznikami czasu na podstawie danych słowo-po-słowie.
    
    Args:
        words_data: Lista słów z danymi timing z ElevenLabs API
        segment_start_time: Czas rozpoczęcia segmentu w oryginalnym wideo (w sekundach)
        timestamp_interval: Interwał znaczników czasu (w sekundach)
        sentence_timestamps: Czy dodawać znaczniki czasu na początku nowych zdań
        
    Returns:
        str: Transkrypt z granularnymi znacznikami czasu
    """
    if not words_data:
        return "Brak danych do przetworzenia."
    
    def format_time(seconds):
        """Formatuj czas w formacie [HH:MM:SS] lub [MM:SS]"""
        total_seconds = int(segment_start_time + seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
        else:
            return f"[{minutes:02d}:{secs:02d}]"
    
    def is_sentence_end(text):
        """Sprawdź czy tekst kończy zdanie"""
        return text.strip().endswith('.') or text.strip().endswith('!') or text.strip().endswith('?')
    
    result_lines = []
    current_speaker = None
    current_text = ""
    current_line_start_time = None
    last_timestamp_time = 0
    sentence_ended = False  # Start with False, first timestamp will be added by speaker change
    
    for i, token in enumerate(words_data):
        token_type = token.get("type", "word")
        speaker_id = token.get("speaker_id", "speaker_0")
        text = token.get("text", "")
        start_time = token.get("start", 0)
        
        if current_speaker != speaker_id:
            if current_text.strip():
                timestamp = format_time(current_line_start_time)
                result_lines.append(f"{timestamp} - {current_speaker}] {current_text.strip()}")
                current_text = ""
            
            current_speaker = speaker_id
            current_line_start_time = start_time
            last_timestamp_time = start_time
            timestamp = format_time(start_time)
            current_text = f"[{timestamp} - {speaker_id}] "
            sentence_ended = False 
        
        elif sentence_timestamps and sentence_ended and token_type == "word" and text.strip():
            if (start_time - last_timestamp_time) >= 2:
                timestamp = format_time(start_time)
                current_text += f" {timestamp} "
                last_timestamp_time = start_time
            sentence_ended = False
        
        elif current_line_start_time is not None and (start_time - last_timestamp_time) >= timestamp_interval:
            timestamp = format_time(start_time)
            current_text += f" {timestamp} "
            last_timestamp_time = start_time
        
        if token_type == "word":
            current_text += text
            if is_sentence_end(text):
                sentence_ended = True
        elif token_type == "spacing":
            current_text += text
    
    if current_text.strip():
        if current_speaker and current_line_start_time is not None:
            if not current_text.strip().startswith('['):
                timestamp = format_time(current_line_start_time)
                result_lines.append(f"[{timestamp} - {current_speaker}] {current_text.strip()}")
            else:
                result_lines.append(current_text.strip())
        else:
            result_lines.append(current_text.strip())
    
    return "\n".join(result_lines)


def transcribe_audio(audio_path, elevenlabs_api_key=None, return_raw_data=False):
    """
    Konwertuje audio na tekst za pomocą ElevenLabs Scribe V1, włączając diariyzację i znaczniki czasu.

    Args:
        audio_path: Ścieżka do pliku audio.
        elevenlabs_api_key: Klucz API ElevenLabs.
        return_raw_data: Czy zwrócić też surowe dane słowo-po-słowie dla granularnych znaczników

    Returns:
        Szczegółowy transkrypt w formacie tekstowym, lub tuple (transkrypt, raw_data) jeśli return_raw_data=True
    """
    if not elevenlabs_api_key:
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')

    if not elevenlabs_api_key:
        logger.warning("Klucz API ElevenLabs nie jest ustawiony, nie można przeprowadzić rozpoznawania mowy")
        return "Nie przeprowadzono rozpoznawania mowy z powodu braku klucza API."

    try:
        audio_size = os.path.getsize(audio_path)
        logger.info(f"Rozmiar pliku audio {audio_path}: {audio_size} bajtów")

        if audio_size < 1024: # Pliki mniejsze niż 1KB są prawdopodobnie puste/nieprawidłowe
            logger.warning(f"Plik audio {audio_path} jest zbyt mały ({audio_size} bajtów), prawdopodobnie jest cichy lub uszkodzony. Pomijanie transkrypcji.")
            return "Plik audio był pusty lub uszkodzony."

        try:
            probe_result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', audio_path
            ], capture_output=True, text=True, timeout=10)

            if probe_result.returncode == 0:
                probe_data = json.loads(probe_result.stdout)
                audio_duration = float(probe_data.get('format', {}).get('duration', 0))
                audio_bitrate = probe_data.get('format', {}).get('bit_rate', 'unknown')

                logger.info(f"Właściwości audio - czas trwania: {audio_duration:.2f}s, bitrate: {audio_bitrate}")

                if audio_duration < 1.0:
                    logger.warning(f"Plik audio {audio_path} jest bardzo krótki ({audio_duration:.2f}s), prawdopodobnie nie zawiera mowy.")
                    return "Plik audio był zbyt krótki."
            else:
                logger.warning(f"Nie udało się sprawdzić właściwości pliku audio: {probe_result.stderr}")
        except Exception as e:
            logger.warning(f"Błąd podczas sprawdzania właściwości audio: {e}")

        models_to_try = ['scribe_v1']
        transcription_result = None
        raw_words_data = None

        for model in models_to_try:
            try:
                logger.info(f"Próbuję model {model} dla {audio_path}...")

                url = "https://api.elevenlabs.io/v1/speech-to-text"
                headers = {"xi-api-key": elevenlabs_api_key}

                with open(audio_path, 'rb') as f:
                    files = {'file': (os.path.basename(audio_path), f, 'audio/mpeg')}

                    data = {
                        'model_id': model,
                        'diarize': 'true',
                        'timestamps_granularity': 'word' 
                    }

                    logger.info(f"Wysyłanie {audio_path} do ElevenLabs Scribe ({model}) z HTTP API...")
                    logger.info(f"Rozmiar pliku: {audio_size} bajtów")
                    response = requests.post(url, headers=headers, files=files, data=data, timeout=300)

                logger.info(f"Otrzymano odpowiedź od ElevenLabs ({model}): status={response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Odpowiedź ElevenLabs ({model}): {json.dumps(result, indent=2, ensure_ascii=False)}")

                    if isinstance(result, dict) and "words" in result and result["words"]:
                        raw_words_data = result["words"]
                        
                        formatted_transcript = []
                        current_speaker = None
                        current_segment_text = ""
                        current_segment_start = None
                        
                        for token in result["words"]:
                            token_type = token.get("type", "word")
                            speaker_id = token.get("speaker_id", "speaker_0")
                            text = token.get("text", "")
                            start_time = token.get("start", 0)
                            
                            if current_speaker != speaker_id and current_segment_text.strip():
                                formatted_line = f"[{time.strftime('%H:%M:%S', time.gmtime(current_segment_start))} - {current_speaker}] {current_segment_text.strip()}"
                                formatted_transcript.append(formatted_line)
                                current_segment_text = ""
                            
                            if current_speaker != speaker_id:
                                current_speaker = speaker_id
                                current_segment_start = start_time
                            
                            if token_type == "word":
                                current_segment_text += text
                            elif token_type == "spacing":
                                current_segment_text += text
                        
                        if current_segment_text.strip():
                            formatted_line = f"[{time.strftime('%H:%M:%S', time.gmtime(current_segment_start))} - {current_speaker}] {current_segment_text.strip()}"
                            formatted_transcript.append(formatted_line)
                        
                        final_text = "\n".join(formatted_transcript)
                        if final_text.strip():
                            logger.info(f"ElevenLabs Scribe ({model}) HTTP zakończone sukcesem. Długość transkryptu: {len(final_text)} znaków")
                            transcription_result = final_text
                            break 
                        else:
                            logger.warning(f"Transkrypcja ({model}) HTTP zakończona, ale transkrypt jest pusty mimo {len(result.get('words', []))} tokenów")
                            transcription_result = "W tym segmencie nie wykryto mowy."
                            continue
                    elif isinstance(result, list) and len(result) > 0:
                        raw_words_data = result
                        
                        formatted_transcript = []
                        current_speaker = None
                        current_segment_text = ""
                        current_segment_start = None
                        
                        for token in result:
                            token_type = token.get("type", "word")
                            speaker_id = token.get("speaker_id", "speaker_0")
                            text = token.get("text", "")
                            start_time = token.get("start", 0)
                            
                            if current_speaker != speaker_id and current_segment_text.strip():
                                formatted_line = f"[{time.strftime('%H:%M:%S', time.gmtime(current_segment_start))} - {current_speaker}] {current_segment_text.strip()}"
                                formatted_transcript.append(formatted_line)
                                current_segment_text = ""
                            
                            if current_speaker != speaker_id:
                                current_speaker = speaker_id
                                current_segment_start = start_time
                            
                            if token_type == "word":
                                current_segment_text += text
                            elif token_type == "spacing":
                                current_segment_text += text
                        
                        if current_segment_text.strip():
                            formatted_line = f"[{time.strftime('%H:%M:%S', time.gmtime(current_segment_start))} - {current_speaker}] {current_segment_text.strip()}"
                            formatted_transcript.append(formatted_line)
                        
                        final_text = "\n".join(formatted_transcript)
                        if final_text.strip():
                            logger.info(f"ElevenLabs Scribe ({model}) HTTP zakończone sukcesem. Długość transkryptu: {len(final_text)} znaków")
                            transcription_result = final_text
                            break
                        else:
                            logger.warning(f"Transkrypcja ({model}) HTTP zakończona, ale transkrypt jest pusty mimo {len(result)} tokenów")
                            transcription_result = "W tym segmencie nie wykryto mowy."
                            continue 
                    elif "segments" in result and result["segments"]:
                        formatted_transcript = []
                        for segment in result["segments"]:
                            speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
                            start_time = segment.get("start", 0)

                            line = f"[{time.strftime('%H:%M:%S', time.gmtime(start_time))} - {speaker}] {segment.get('text', '').strip()}"
                            formatted_transcript.append(line)

                        final_text = "\n".join(formatted_transcript)
                        logger.info(f"ElevenLabs Scribe ({model}) HTTP zakończone sukcesem. Długość transkryptu: {len(final_text)} znaków")
                        transcription_result = final_text
                        break 
                    else:
                        logger.warning(f"Transkrypcja ({model}) HTTP zakończona, ale nie wykryto mowy w {audio_path}. Pełna odpowiedź API: {json.dumps(result, indent=2, ensure_ascii=False)}")
                        transcription_result = "W tym segmencie nie wykryto mowy."
                        continue 
                else:
                    error_details = response.text
                    logger.error(f"Rozpoznawanie mowy przez ElevenLabs ({model}) nie powiodło się: {response.status_code}. Szczegóły: {error_details}")
                    transcription_result = f"Rozpoznawanie mowy nie powiodło się: {response.status_code}, {error_details}"
                    continue

            except Exception as e:
                logger.error(f"Wystąpił wyjątek podczas rozpoznawania mowy przez ElevenLabs ({model}) HTTP: {e}")
                import traceback
                logger.error(f"Stos błędów: {traceback.format_exc()}")
                transcription_result = f"Błąd krytyczny podczas rozpoznawania mowy: {str(e)}"
                continue 

        if return_raw_data:
            return transcription_result, raw_words_data
        return transcription_result

    except Exception as e:
        logger.error(f"Wystąpił wyjątek podczas rozpoznawania mowy przez ElevenLabs: {e}")
        import traceback
        logger.error(f"Stos błędów: {traceback.format_exc()}")
        error_msg = f"Błąd krytyczny podczas rozpoznawania mowy: {str(e)}"
        if return_raw_data:
            return error_msg, None
        return error_msg


def analyze_video_content(video_path, openai_api_key=None, frames_count=1, timestamps=False, segment_start_time=0):
    """
    Analizuje zawartość wideo za pomocą modelu OpenAI Vision.

    Args:
        video_path: Ścieżka do pliku wideo.
        segment_start_time: Czas rozpoczęcia segmentu w oryginalnym wideo (w sekundach).
        openai_api_key: Klucz API OpenAI.
        frames_count: Liczba klatek kluczowych do wyodrębnienia, domyślnie 1.
        timestamps: Czy dodać znaczniki czasu do wyodrębnianych klatek.

    Returns:
        Tekst z wynikami analizy.
    """
    if not openai_api_key:
        openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        logger.warning("Klucz API OpenAI nie jest ustawiony, nie można przeprowadzić analizy zawartości wideo")
        return "Nie przeprowadzono analizy zawartości wideo"

    if frames_count == 1:
        frames_count = int(os.getenv('FRAMES_PER_SEGMENT', '1'))

    try:
        from openai import OpenAI
        import base64

        client = OpenAI(api_key=openai_api_key)

        video_name = os.path.basename(video_path)
        segment_id = os.path.splitext(video_name)[0]

        frame_paths = []
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
                 'default=noprint_wrappers=1:nokey=1', video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            duration = float(result.stdout.strip())

            if frames_count == 1:
                frame_path = os.path.join(os.path.dirname(video_path),
                                          f"{os.path.splitext(os.path.basename(video_path))[0]}_keyframe.jpg")
                middle_time = duration / 2

                if timestamps:
                    base_name = os.path.basename(video_path)
                    
                    actual_time = segment_start_time + middle_time
                    
                    hours = int(actual_time // 3600)
                    minutes = int((actual_time % 3600) // 60)
                    seconds = int(actual_time % 60)
                    
                    if hours > 0:
                        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        time_str = f"{minutes:02d}:{seconds:02d}"
                    
                    time_str_escaped = time_str.replace(":", "\\:")
                    text = f"{base_name} {time_str_escaped}"
                    draw_filter = (
                        f"drawtext=fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5"
                        f":text='{text}':x=w-tw-10:y=h-th-10"
                    )
                    
                    subprocess.run([
                        'ffmpeg', '-y', '-ss', str(middle_time),
                        '-i', video_path, '-vframes', '1',
                        '-vf', draw_filter, '-q:v', '2', frame_path
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                else:
                    subprocess.run([
                        'ffmpeg', '-y', '-ss', str(middle_time),
                        '-i', video_path, '-vframes', '1',
                        '-q:v', '2', frame_path
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                frame_paths = [frame_path]
                logger.info(f"Pomyślnie wyodrębniono klatkę kluczową: {frame_path}")
            else:
                interval = duration / (frames_count + 1)  # +1, aby uniknąć ostatniej klatki na samym końcu

                for i in range(frames_count):
                    frame_time = interval * (i + 1)
                    frame_path = os.path.join(os.path.dirname(video_path),
                                              f"{os.path.splitext(os.path.basename(video_path))[0]}_keyframe_{i+1}.jpg")

                    if timestamps:
                        base_name = os.path.basename(video_path)
                        
                        actual_time = segment_start_time + frame_time
                        
                        hours = int(actual_time // 3600)
                        minutes = int((actual_time % 3600) // 60)
                        seconds = int(actual_time % 60)
                        
                        if hours > 0:
                            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        else:
                            time_str = f"{minutes:02d}:{seconds:02d}"
                        
                        time_str_escaped = time_str.replace(":", "\\:")
                        text = f"{base_name} {time_str_escaped}"
                        draw_filter = (
                            f"drawtext=fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5"
                            f":text='{text}':x=w-tw-10:y=h-th-10"
                        )
                        
                        subprocess.run([
                            'ffmpeg', '-y', '-ss', str(frame_time),
                            '-i', video_path, '-vframes', '1',
                            '-vf', draw_filter, '-q:v', '2', frame_path
                        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        subprocess.run([
                            'ffmpeg', '-y', '-ss', str(frame_time),
                            '-i', video_path, '-vframes', '1',
                            '-q:v', '2', frame_path
                        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    frame_paths.append(frame_path)
                    logger.info(f"Pomyślnie wyodrębniono klatkę kluczową {i+1}/{frames_count}: {frame_path}")

                logger.info(f"Pomyślnie wyodrębniono {frames_count} klatek kluczowych")

        except Exception as e:
            logger.error(f"Nie udało się wyodrębnić klatek kluczowych: {e}")
            return f"Nie udało się wyodrębnić klatek kluczowych: {str(e)}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Przeanalizuj dokładnie załączony obraz będący slajdem prezentacji i wypisz wszystkie możliwe informacje w formie szczegółowej listy punktowanej. Uwzględnij:

Tytuł i nagłówki – rozpoznaj główny nagłówek slajdu.

Struktura treści – odczytaj wszystkie numery, nazwy punktów agendy, kolejność i język (w tym polskie znaki).

Logo i branding – wskaż obecność logotypu (litera W = logo firmy Acme).

Dodatkowe szczegóły – np. wszelkie napisy w stopce, znaki wodne, metadane lub informacje techniczne widoczne na obrazie.

Interpretacja treści – wyjaśnij, czego dotyczy agenda (np. nowe produkty, technologie, funkcje).

Zadbaj, aby wynik był pełną, precyzyjną i uporządkowaną listą punktów, nie pomijając żadnych detali, nawet drobnych elementów graficznych."""
                    }
                ]
            }
        ]

        for frame_path in frame_paths:
            with open(frame_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            max_tokens=1000
        )

        analysis = response.choices[0].message.content
        logger.info(f"Analiza przez OpenAI Vision zakończona sukcesem, długość tekstu: {len(analysis)}")

        formatted_analysis = f"""
Analiza wizualna fragmentu wideo (ID: {segment_id}):

{analysis}
        """

        return formatted_analysis

    except Exception as e:
        logger.error(f"Błąd podczas analizy zawartości wideo przez OpenAI Vision: {e}")
        import traceback
        logger.error(f"Stos błędów: {traceback.format_exc()}")
        return f"Błąd podczas analizy zawartości wideo: {str(e)}"





def process_video_segment(segment_info, elevenlabs_api_key=None, openai_api_key=None, skip_video_analysis=False, frames_per_segment=1, segment_duration=200, video_path=None, timestamps=False):
    """
    Przetwarza pojedynczy segment wideo, w tym transkrypcję mowy na tekst i analizę zawartości wideo.

    Args:
        segment_info: Informacje o segmencie wideo.
        elevenlabs_api_key: Klucz API ElevenLabs.
        openai_api_key: Klucz API OpenAI.
        skip_video_analysis: Czy pominąć analizę zawartości wideo.
        frames_per_segment: Liczba klatek kluczowych do wyodrębnienia na segment.
        segment_duration: Długość segmentu w sekundach.
        video_path: Ścieżka do oryginalnego pliku wideo.
        timestamps: Czy dodać znaczniki czasu do klatek wideo.

    Returns:
        Słownik z wynikami przetwarzania.
    """
    result = {
        'segment_info': segment_info,
        'transcript': None,
        'visual_analysis': None
    }

    if 'mp3_path' in segment_info:
        logger.info(f"Rozpoczynam przetwarzanie audio: {segment_info['mp3_path']}")
        
        transcript, raw_words = transcribe_audio(segment_info['mp3_path'], elevenlabs_api_key, return_raw_data=True)
        result['transcript'] = transcript
        
        if raw_words:
            logger.info("Generuję transkrypt z granularnymi znacznikami czasu...")
            segment_start_time = segment_info.get('start_time', 0)
            granular_transcript = create_granular_transcript(
                raw_words, 
                segment_start_time, 
                timestamp_interval=10, 
                sentence_timestamps=True
            )
            result['granular_transcript'] = granular_transcript
            logger.info(f"Wygenerowano granularny transkrypt o długości {len(granular_transcript)} znaków")
        else:
            result['granular_transcript'] = transcript
    if 'mp4_path' in segment_info and not skip_video_analysis:
        logger.info(f"Rozpoczynam analizę zawartości wideo: {segment_info['mp4_path']}, wyodrębnianie {frames_per_segment} klatek")
        visual_analysis = analyze_video_content(segment_info['mp4_path'], openai_api_key, frames_per_segment, timestamps, segment_info.get('start_time', 0))
        result['visual_analysis'] = visual_analysis
    elif skip_video_analysis:
        logger.info("Pominięto analizę zawartości wideo")
        result['visual_analysis'] = "Pominięto analizę zawartości wideo (konto API może mieć zaległości w płatnościach lub problemy z uprawnieniami)"

    output_dir = os.path.dirname(segment_info.get('mp4_path', ''))
    if output_dir:
        segment_name = os.path.splitext(os.path.basename(segment_info.get('mp4_path', '')))[0]
        json_result_path = os.path.join(output_dir, f"{segment_name}_result.json")
        with open(json_result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        result['json_result_path'] = json_result_path
        logger.info(f"Wyniki JSON segmentu zostały zapisane do: {json_result_path}")

    return result


def main():
    global RAGFLOW_API_URL, RAGFLOW_API_KEY
    
    load_environment_config()
    
    video_paths = [
        '/home/zy/ragflow/1101732139-1-208.mp4'
    ]
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Konfiguracja ścieżek wideo zakończona: {len(video_paths)} plików wideo")

    output_dir = os.getenv('OUTPUT_DIR', './video_segments')
    segment_duration = int(os.getenv('SEGMENT_DURATION', '200'))
    dataset_name = os.getenv('DATASET_NAME', 'Video Segments Dataset')
    frames_per_segment = int(os.getenv('FRAMES_PER_SEGMENT', '1'))

    skip_video_analysis = os.getenv('SKIP_VIDEO_ANALYSIS', 'false').lower() == 'true'
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Konfiguracja funkcji: Analiza wideo {'pominięta' if skip_video_analysis else 'włączona'}, {frames_per_segment} klatek na segment")

    import logging
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Konfiguracja RAGFlow zakończona, adres usługi: {RAGFLOW_API_URL}")

    elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Konfiguracja kluczy API dużych modeli zakończona")

    timestamps_enabled = False

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Przetwarza wideo i przesyła do RAGFlow')
        parser.add_argument('--video', type=str, help='Ścieżka do pliku wideo')
        parser.add_argument('--output-dir', type=str, default='./video_segments', help='Katalog wyjściowy')
        parser.add_argument('--segment-duration', type=int, default=200, help='Czas trwania segmentu wideo (w sekundach)')
        parser.add_argument('--frames-per-segment', type=int, default=None, help='Liczba klatek kluczowych do wyodrębnienia na segment')
        parser.add_argument('--dataset-name', type=str, default=None, help='Nazwa zbioru danych RAGFlow')
        parser.add_argument('--ragflow-api-url', type=str, default=None, help='URL API RAGFlow')
        parser.add_argument('--ragflow-api-key', type=str, default=None, help='Klucz API RAGFlow')
        parser.add_argument('--enable-video-analysis', action='store_true',
                            help='Włącz analizę zawartości wideo (domyślnie pominięta, wymaga ważnego klucza API)')
        parser.add_argument('--timestamps', action='store_true',
                            help='Dodaj znaczniki czasu do dolnego prawego rogu każdej klatki wideo')

        args = parser.parse_args()

        if args.video:
            video_paths = [args.video]

        if args.timestamps:
            timestamps_enabled = True

        if args.output_dir:
            output_dir = args.output_dir
        if args.segment_duration:
            segment_duration = args.segment_duration
        if args.frames_per_segment:
            frames_per_segment = args.frames_per_segment
        if args.dataset_name:
            dataset_name = args.dataset_name
        if args.ragflow_api_url or args.ragflow_api_key:
            if args.ragflow_api_url:
                RAGFLOW_API_URL = args.ragflow_api_url
            if args.ragflow_api_key:
                RAGFLOW_API_KEY = args.ragflow_api_key

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Rozpoczynam przetwarzanie wideo...")

    if not video_paths:
        logger.error("Nie podano ścieżki do wideo, program zostanie zakończony")
        return

    video_path_raw = video_paths[0]
    video_path, video_options = parse_video_path_with_params(video_path_raw)
    
    timestamps_enabled = timestamps_enabled or video_options.get("timestamps", False)
    
    if timestamps_enabled:
        logger.info("✅ Funkcja znaczników czasu jest włączona")
    else:
        logger.info("ℹ️ Funkcja znaczników czasu jest wyłączona")

    logger.info(f"Rozpoczynam przetwarzanie wideo: {video_path}")
    segments = segment_video(video_path, output_dir, segment_duration, timestamps_enabled)

    if not segments:
        logger.error("Segmentacja wideo nie powiodła się, program zostanie zakończony")
        return

    results = []
    for segment in segments:
        logger.info(f"Rozpoczynam przetwarzanie segmentu {segment['index'] + 1}/{len(segments)}")
        result = process_video_segment(segment, elevenlabs_api_key, openai_api_key, skip_video_analysis, frames_per_segment, segment_duration=segment_duration, video_path=video_path, timestamps=timestamps_enabled)
        results.append(result)

    logger.info("Rozpoczynam generowanie kompleksowego raportu z wszystkich segmentów...")
    comprehensive_report_path = create_comprehensive_video_report(
        results, 
        video_path, 
        output_dir,
        segment_duration, 
        frames_per_segment
    )
    
    if comprehensive_report_path:
        logger.info(f"Kompleksowy raport został zapisany do: {comprehensive_report_path}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Kompleksowy raport wygenerowany: {comprehensive_report_path}")
    else:
        logger.error("Nie udało się wygenerować kompleksowego raportu")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ Błąd podczas generowania kompleksowego raportu")

    if RAGFLOW_API_KEY:
        logger.info(f"Rozpoczynam przesyłanie wyników do RAGFlow, nazwa zbioru danych: {dataset_name}")

        dataset_id = get_dataset_by_name(dataset_name)

        if not dataset_id:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Zbiór danych nie istnieje, próba utworzenia nowego zbioru danych...")
            dataset_id = create_ragflow_dataset(dataset_name)
            if not dataset_id:
                logger.error("Nie udało się utworzyć zbioru danych RAGFlow, nie można kontynuować")
                return
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pomyślnie utworzono zbiór danych RAGFlow")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Znaleziono istniejący zbiór danych, ID: {dataset_id}, zostanie on użyty bezpośrednio")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔍 Weryfikacja konfiguracji grafu wiedzy zbioru danych...")
        config_result = verify_dataset_graphrag_config(dataset_id)
        if config_result["success"]:
            if config_result["enabled"]:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Funkcja grafu wiedzy dla zbioru danych jest poprawnie włączona")
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Ostrzeżenie: Funkcja grafu wiedzy dla zbioru danych nie jest włączona, co może wpłynąć na działanie asystenta")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ Nie można zweryfikować konfiguracji grafu wiedzy zbioru danych: {config_result.get('error')}")

        document_ids = []
        
        if comprehensive_report_path and os.path.exists(comprehensive_report_path):
            comprehensive_report_name = os.path.basename(comprehensive_report_path)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Przesyłanie kompleksowego raportu: {comprehensive_report_name}")
            comprehensive_doc_id = upload_file_to_ragflow(dataset_id, comprehensive_report_name, comprehensive_report_path, auto_parse=False)
            if comprehensive_doc_id:
                update_success = update_document_parser_config(dataset_id, comprehensive_doc_id)
                if update_success:
                    document_ids.append(comprehensive_doc_id)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Kompleksowy raport (ID: {comprehensive_doc_id}) został dodany do kolejki parsowania")
                else:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ostrzeżenie: Aktualizacja konfiguracji parsowania dla kompleksowego raportu nie powiodła się")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Kompleksowy raport nie został wygenerowany, pomijanie przesyłania")

        if document_ids:
            logger.info(f"Rozpoczynam wsadowe uruchamianie parsowania wszystkich dokumentów, liczba dokumentów: {len(document_ids)}")
            success_count = 0
            triggered_doc_ids = []
            for doc_id in document_ids:
                parse_success = parse_documents(dataset_id, [doc_id])
                if parse_success:
                    success_count += 1
                    triggered_doc_ids.append(doc_id)
                    logger.info(f"Uruchomienie parsowania dokumentu {doc_id} powiodło się")
                else:
                    logger.warning(f"Uruchomienie parsowania dokumentu {doc_id} nie powiodło się")
                time.sleep(0.5)

            logger.info(f"Uruchomiono parsowanie dla {success_count}/{len(document_ids)} dokumentów")

            if triggered_doc_ids:
                logger.info("Rozpoczynam oczekiwanie na zakończenie parsowania dokumentów...")
                parsing_result = wait_for_documents_parsing(
                    dataset_id,
                    triggered_doc_ids,
                    check_interval=30
                )

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Statystyki zakończenia parsowania dokumentów:")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - Całkowita liczba dokumentów: {parsing_result['total']}")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - Sparsowano pomyślnie: {parsing_result['completed']}")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - Błędy parsowania: {parsing_result['failed']}")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - Czas parsowania: {parsing_result['elapsed_time']:.1f} sekund")

                if parsing_result.get("completed", 0) > 0:
                    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🎉 Parsowanie dokumentów zakończone!")
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 💡 Teraz możesz utworzyć asystenta wideo, uruchamiając następujące polecenie:")
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]    python create_video_assistant.py {dataset_id}")
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📋 ID zbioru danych: {dataset_id}")
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🌐 Interfejs RAGFlow: {RAGFLOW_API_URL}")

                if parsing_result['failed'] > 0:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ID dokumentów z błędami: {parsing_result['failed_ids']}")
            else:
                logger.warning("Nie udało się uruchomić parsowania żadnego dokumentu")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Wszystkie operacje zakończone!")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Całkowity czas: {total_time:.2f} sekund ({total_time / 60:.2f} minut)")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Przetwarzanie zakończone, przesłano {len(document_ids)} dokumentów do RAGFlow")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ID zbioru danych RAGFlow: {dataset_id}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Adres interfejsu RAGFlow: {RAGFLOW_API_URL}")
    else:
        logger.warning("Klucz API RAGFlow nie jest ustawiony, pomijanie kroku przesyłania")
        logger.info(f"Przetwarzanie zakończone, wyniki zapisane w: {output_dir}")


if __name__ == '__main__':
    main()