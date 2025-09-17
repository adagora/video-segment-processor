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
import tempfile
from pathlib import Path
import argparse
from dotenv import load_dotenv


# Ignoruj ostrzeżenia
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)


# Ustaw format logów
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Konfiguracja API RAGFlow - ładowana z zmiennych środowiskowych
RAGFLOW_API_URL = None  # Ładowane z env
RAGFLOW_API_KEY = None  # Ładowane z env


def load_environment_config():
    """Załaduj konfigurację środowiska"""
    # Spróbuj załadować plik .env
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        logger.info(f"Załadowano plik zmiennych środowiskowych: {env_file}")
    else:
        # Jeśli nie ma .env, spróbuj config.env
        config_file = os.path.join(os.path.dirname(__file__), 'config.env')
        if os.path.exists(config_file):
            load_dotenv(config_file)
            logger.info(f"Załadowano plik konfiguracyjny: {config_file}")
        else:
            logger.warning("Nie znaleziono .env ani config.env, używane będą zmienne systemowe")
    
    # Załaduj konfigurację do zmiennych globalnych
    global RAGFLOW_API_URL, RAGFLOW_API_KEY
    RAGFLOW_API_URL = os.getenv('RAGFLOW_API_URL', 'http://localhost:9380')
    RAGFLOW_API_KEY = os.getenv('RAGFLOW_API_KEY')
    
    # Sprawdź wymagane ustawienia
    if not RAGFLOW_API_KEY:
        logger.warning("Nie ustawiono RAGFLOW_API_KEY, funkcje RAGFlow będą niedostępne")
    
    logger.info(f"Konfiguracja RAGFlow załadowana: URL={RAGFLOW_API_URL}")


def get_dataset_by_name(name):
    """Pobierz ID zbioru danych RAGFlow na podstawie nazwy"""
    headers = {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}"
    }

    # Pobierz wszystkie zbiory danych
    response = requests.get(
        f"{RAGFLOW_API_URL}/api/v1/datasets?name={name}",
        headers=headers
    )

    if response.status_code == 200:
        result = response.json()
        if result.get("code") == 0:
            datasets = result.get("data", [])
            # Szukaj zbioru danych o podanej nazwie
            for dataset in datasets:
                if dataset.get("name") == name:
                    logger.info(f"Znaleziono istniejący zbiór danych: {name}, ID: {dataset.get('id')}")
                    return dataset.get('id')

    logger.info(f"Nie znaleziono zbioru danych o nazwie {name}")
    return None


# Stała konfiguracja grafu wiedzy (GraphRAG), zapewnia spójność
GRAPHRAG_CONFIG = {
    "use_graphrag": True,  # Włącz funkcję grafu wiedzy
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

    # Konfiguracja zawiera ustawienia grafu wiedzy (graphrag)
    payload = {
        "name": name,
        "description": description,
    "chunk_method": "naive",  # Użyj ogólnej metody dzielenia
        "parser_config": {
            "chunk_token_num": 512,  # Zmniejsz liczbę tokenów, aby przyspieszyć przetwarzanie
            "delimiter": "\n",
            "filename_embd_weight": 0.1,
            "task_page_size": 5,  # Zmniejsz rozmiar strony zadania, aby uniknąć przekroczenia czasu
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


                # Sprawdź kluczowe ustawienia
                chunk_method = dataset_data.get("chunk_method", "unknown")
                parser_id = dataset_data.get("parser_id", "unknown")
                is_graphrag_enabled = graphrag_config.get("use_graphrag", False)

                # Prawdziwy warunek aktywacji grafu wiedzy
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
    # Użyj API List documents, aby pobrać informacje o konkretnym dokumencie po ID
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

    # Sprawdź czy każdy dokument ma chunk
        for doc_id in document_ids:
            if doc_id in completed_docs or doc_id in failed_docs:
                continue

            status_info = check_document_status(dataset_id, doc_id)
            chunk_count = status_info.get("chunk_count", 0)

            # Proste sprawdzenie: jeśli są chunk, to zakończono
            if chunk_count > 0:
                completed_docs.add(doc_id)
                print(f"✅ Dokument {doc_id} przetworzony (liczba chunków: {chunk_count})")
            elif status_info.get("status") in ["ERROR", "NOT_FOUND"]:
                failed_docs.add(doc_id)
                print(f"❌ Dokument {doc_id} nie został przetworzony")


        # Wyświetl aktualny status
        remaining = len(document_ids) - len(completed_docs) - len(failed_docs)
        print(f"⏱️  Czas oczekiwania: {format_time(elapsed_time)} | "
              f"✅ Zakończone: {len(completed_docs)} | "
              f"❌ Nieudane: {len(failed_docs)} | "
              f"⏳ Oczekujące: {remaining}")

        # Jeśli wszystkie dokumenty są przetworzone, zakończ pętlę
        if len(completed_docs) + len(failed_docs) == len(document_ids):
            print(f"\n{'=' * 60}")
            print("🎉 Wszystkie dokumenty zostały przetworzone!")
            print(f"{'=' * 60}")
            break

        # Poczekaj przed kolejnym sprawdzeniem
        print(f"Czekam {check_interval} sekund przed kolejnym sprawdzeniem...")
        time.sleep(check_interval)

    # Zwróć statystyki przetwarzania
    result = {
        "total": len(document_ids),
        "completed": len(completed_docs),
        "failed": len(failed_docs),
        "completed_ids": list(completed_docs),
        "failed_ids": list(failed_docs),
        "elapsed_time": time.time() - start_time
    }

    # Wyświetl końcowe statystyki
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

    # Uproszczona logika: wyślij wszystkie ID dokumentów, zgodnie z process_videos_to_ragflow.py
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
            timeout=60  # Zwiększono limit czasu
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

    # Ustaw konfigurację parsera dokumentu, włącz GraphRAG
    payload = {
    "chunk_method": "naive",  # Użyj ogólnej metody dzielenia
        "parser_config": {
            "chunk_token_num": 512,  # Zmniejsz liczbę tokenów, aby przyspieszyć przetwarzanie
            "delimiter": "\n",
            "filename_embd_weight": 0.1,
            "task_page_size": 5,  # Zmniejsz rozmiar strony zadania, aby uniknąć przekroczenia czasu
            "graphrag": GRAPHRAG_CONFIG  # Użyj spójnej konfiguracji GraphRAG
        }
    }

    try:
    # Wyślij żądanie aktualizacji
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
                # Nawet jeśli aktualizacja się nie powiedzie, kontynuuj
                return True
        else:
            logger.error(f"Nie udało się zaktualizować konfiguracji parsera dokumentu, kod statusu: {response.status_code}")
            # Nawet jeśli aktualizacja się nie powiedzie, kontynuuj
            return True
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji konfiguracji parsera dokumentu: {e}")
        # Nawet jeśli wystąpi błąd, kontynuuj
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

    # Ustal MIME type pliku
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

    # Wyślij plik jako multipart/form-data
    try:
        with open(file_path, 'rb') as f:
            files = {
                'file': (file_name, f, mime_type)
            }

            # Jeśli określono typ pliku jako binary, dodaj parametr file_type
            params = {}
            if file_type == 'binary':
                params['file_type'] = 'binary'

            logger.info(f"Rozpoczynam wysyłanie pliku {file_name} do zbioru danych {dataset_id}")

            response = requests.post(
                f"{RAGFLOW_API_URL}/api/v1/datasets/{dataset_id}/documents",
                headers=headers,
                files=files,
                params=params,
                timeout=60  # Zwiększono limit czasu
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

            # Jeśli auto_parse, natychmiast wyzwól przetwarzanie dokumentu
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


def segment_video(video_path, output_dir, segment_duration=200):
    """
    Segmentuje wideo i zapisuje do określonego katalogu.

    Args:
        video_path: Ścieżka do pliku wideo.
        output_dir: Katalog wyjściowy.
        segment_duration: Czas trwania każdego segmentu wideo (w sekundach).

    Returns:
        Lista słowników z informacjami o segmentach. Każdy słownik zawiera klucze
        takie jak 'mp4_path', 'mp3_path', 'start_time', 'end_time'.
    """
    # Upewnij się, że katalog wyjściowy istnieje
    os.makedirs(output_dir, exist_ok=True)

    # Pobierz całkowity czas trwania wideo
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

    # Oblicz liczbę segmentów
    num_segments = int(duration / segment_duration) + (1 if duration % segment_duration > 0 else 0)
    logger.info(f"Podziel wideo na {num_segments} segmentów, każdy po {segment_duration} sekund")

    # Nazwa pliku wideo (bez rozszerzenia)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, duration)

        # Utwórz informacje o segmencie
        segment_info = {
            'index': i,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        }

        # Zbuduj ścieżkę pliku wyjściowego
        mp4_path = os.path.join(output_dir, f"{video_name}_segment_{i}.mp4")
        mp3_path = os.path.join(output_dir, f"{video_name}_segment_{i}.mp3")

        # Wyodrębnij segment wideo
        try:
            logger.info(f"Wyodrębnianie segmentu wideo {i + 1}/{num_segments}: {start_time:.2f}s - {end_time:.2f}s")
            subprocess.run([
                'ffmpeg', '-y', '-ss', str(start_time),
                '-to', str(end_time), '-i', video_path,
                '-c', 'copy', mp4_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            segment_info['mp4_path'] = mp4_path
        except Exception as e:
            logger.error(f"Nie udało się wyodrębnić segmentu wideo: {e}")
            continue

        # Wyodrębnij segment audio
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


def transcribe_audio(audio_path, elevenlabs_api_key=None):
    """
    Konwertuje audio na tekst za pomocą ElevenLabs Scribe V1, włączając diariyzację i znaczniki czasu.

    Args:
        audio_path: Ścieżka do pliku audio.
        elevenlabs_api_key: Klucz API ElevenLabs.

    Returns:
        Szczegółowy transkrypt w formacie tekstowym, lub komunikat o błędzie.
    """
    # Użyj przekazanego klucza API lub pobierz go ze zmiennych środowiskowych
    if not elevenlabs_api_key:
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')

    if not elevenlabs_api_key:
        logger.warning("Klucz API ElevenLabs nie jest ustawiony, nie można przeprowadzić rozpoznawania mowy")
        return "Nie przeprowadzono rozpoznawania mowy z powodu braku klucza API."

    try:
        # Sprawdź, czy plik audio nie jest pusty
        audio_size = os.path.getsize(audio_path)
        logger.info(f"Rozmiar pliku audio {audio_path}: {audio_size} bajtów")

        if audio_size < 1024: # Pliki mniejsze niż 1KB są prawdopodobnie puste/nieprawidłowe
            logger.warning(f"Plik audio {audio_path} jest zbyt mały ({audio_size} bajtów), prawdopodobnie jest cichy lub uszkodzony. Pomijanie transkrypcji.")
            return "Plik audio był pusty lub uszkodzony."

        # Sprawdź właściwości audio za pomocą ffprobe
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

        # Lista modeli do wypróbowania w kolejności (od najbardziej dokładnego do podstawowego)
        models_to_try = ['scribe_v1']
        transcription_result = None

        for model in models_to_try:
            try:
                logger.info(f"Próbuję model {model} dla {audio_path}...")

                # Użyj tylko metody HTTP API
                url = "https://api.elevenlabs.io/v1/speech-to-text"
                headers = {"xi-api-key": elevenlabs_api_key}

                with open(audio_path, 'rb') as f:
                    files = {'file': (os.path.basename(audio_path), f, 'audio/mpeg')}

                    # Ustaw parametry żądania, aby uzyskać szczegółowe dane
                    data = {
                        'model_id': model,
                        'diarize': 'true',  # Włącz diariyzację (identyfikację mówców)
                        'timestamps_granularity': 'word'  # Znaczniki czasu na poziomie słów
                    }

                    logger.info(f"Wysyłanie {audio_path} do ElevenLabs Scribe ({model}) z HTTP API...")
                    logger.info(f"Rozmiar pliku: {audio_size} bajtów")
                    response = requests.post(url, headers=headers, files=files, data=data, timeout=300)

                logger.info(f"Otrzymano odpowiedź od ElevenLabs ({model}): status={response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Odpowiedź ElevenLabs ({model}): {json.dumps(result, indent=2, ensure_ascii=False)}")

                    if isinstance(result, dict) and "words" in result and result["words"]:
                        formatted_transcript = []
                        current_speaker = None
                        current_segment_text = ""
                        current_segment_start = None
                        
                        for token in result["words"]:
                            token_type = token.get("type", "word")
                            speaker_id = token.get("speaker_id", "speaker_0")
                            text = token.get("text", "")
                            start_time = token.get("start", 0)
                            
                            # Jeśli zmienił się mówca, zakończ poprzedni segment
                            if current_speaker != speaker_id and current_segment_text.strip():
                                formatted_line = f"[{time.strftime('%H:%M:%S', time.gmtime(current_segment_start))} - {current_speaker}] {current_segment_text.strip()}"
                                formatted_transcript.append(formatted_line)
                                current_segment_text = ""
                            
                            # Aktualizuj informacje o bieżącym segmencie
                            if current_speaker != speaker_id:
                                current_speaker = speaker_id
                                current_segment_start = start_time
                            
                            # Dodaj tekst do bieżącego segmentu
                            if token_type == "word":
                                current_segment_text += text
                            elif token_type == "spacing":
                                current_segment_text += text
                        
                        # Dodaj ostatni segment
                        if current_segment_text.strip():
                            formatted_line = f"[{time.strftime('%H:%M:%S', time.gmtime(current_segment_start))} - {current_speaker}] {current_segment_text.strip()}"
                            formatted_transcript.append(formatted_line)
                        
                        final_text = "\n".join(formatted_transcript)
                        if final_text.strip():
                            logger.info(f"ElevenLabs Scribe ({model}) HTTP zakończone sukcesem. Długość transkryptu: {len(final_text)} znaków")
                            transcription_result = final_text
                            break  # Sukces, przerywamy pętlę
                        else:
                            logger.warning(f"Transkrypcja ({model}) HTTP zakończona, ale transkrypt jest pusty mimo {len(result.get('words', []))} tokenów")
                            transcription_result = "W tym segmencie nie wykryto mowy."
                            continue  # Próbuj następny model
                    elif isinstance(result, list) and len(result) > 0:
                        # Stary format odpowiedzi ElevenLabs - lista tokenów bezpośrednio
                        formatted_transcript = []
                        current_speaker = None
                        current_segment_text = ""
                        current_segment_start = None
                        
                        for token in result:
                            token_type = token.get("type", "word")
                            speaker_id = token.get("speaker_id", "speaker_0")
                            text = token.get("text", "")
                            start_time = token.get("start", 0)
                            
                            # Jeśli zmienił się mówca, zakończ poprzedni segment
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
                        # Stary format odpowiedzi ElevenLabs - segmenty
                        formatted_transcript = []
                        for segment in result["segments"]:
                            speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
                            start_time = segment.get("start", 0)

                            # Utwórz sformatowany wiersz dla każdego segmentu
                            line = f"[{time.strftime('%H:%M:%S', time.gmtime(start_time))} - {speaker}] {segment.get('text', '').strip()}"
                            formatted_transcript.append(line)

                        final_text = "\n".join(formatted_transcript)
                        logger.info(f"ElevenLabs Scribe ({model}) HTTP zakończone sukcesem. Długość transkryptu: {len(final_text)} znaków")
                        transcription_result = final_text
                        break 
                    else:
                        # Obsługa przypadku, gdy w audio nie wykryto mowy
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

        return transcription_result

    except Exception as e:
        logger.error(f"Wystąpił wyjątek podczas rozpoznawania mowy przez ElevenLabs: {e}")
        import traceback
        logger.error(f"Stos błędów: {traceback.format_exc()}")
        return f"Błąd krytyczny podczas rozpoznawania mowy: {str(e)}"


def analyze_video_content(video_path, openai_api_key=None, frames_count=1):
    """
    Analizuje zawartość wideo za pomocą modelu OpenAI Vision.

    Args:
        video_path: Ścieżka do pliku wideo.
        openai_api_key: Klucz API OpenAI.
        frames_count: Liczba klatek kluczowych do wyodrębnienia, domyślnie 1.

    Returns:
        Tekst z wynikami analizy.
    """
    # Użyj przekazanego klucza API lub pobierz go ze zmiennych środowiskowych
    if not openai_api_key:
        openai_api_key = os.getenv('OPENAI_API_KEY')

    if not openai_api_key:
        logger.warning("Klucz API OpenAI nie jest ustawiony, nie można przeprowadzić analizy zawartości wideo")
        return "Nie przeprowadzono analizy zawartości wideo"

    # Pobierz liczbę klatek ze zmiennych środowiskowych, jeśli nie podano parametru
    if frames_count == 1:
        frames_count = int(os.getenv('FRAMES_PER_SEGMENT', '1'))

    try:
        from openai import OpenAI
        import base64

        client = OpenAI(api_key=openai_api_key)

        # Użyj nazwy pliku wideo jako ID wideo
        video_name = os.path.basename(video_path)
        segment_id = os.path.splitext(video_name)[0]

        # Wyodrębnij klatki kluczowe
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
                # Ekstrakcja pojedynczej klatki: wyodrębnij klatkę ze środka wideo
                frame_path = os.path.join(os.path.dirname(video_path),
                                          f"{os.path.splitext(os.path.basename(video_path))[0]}_keyframe.jpg")
                middle_time = duration / 2

                subprocess.run([
                    'ffmpeg', '-y', '-ss', str(middle_time),
                    '-i', video_path, '-vframes', '1',
                    '-q:v', '2', frame_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                frame_paths = [frame_path]
                logger.info(f"Pomyślnie wyodrębniono klatkę kluczową: {frame_path}")
            else:
                # Ekstrakcja wielu klatek: wyodrębnij równomiernie rozłożone klatki kluczowe
                interval = duration / (frames_count + 1)  # +1, aby uniknąć ostatniej klatki na samym końcu

                for i in range(frames_count):
                    frame_time = interval * (i + 1)
                    frame_path = os.path.join(os.path.dirname(video_path),
                                              f"{os.path.splitext(os.path.basename(video_path))[0]}_keyframe_{i+1}.jpg")

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

        # Użyj API OpenAI Vision do analizy obrazów
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Proszę szczegółowo przeanalizować zawartość tego fragmentu wideo, w tym scenę, postacie, działania, atmosferę i elementy wizualne. Proszę opisać w języku polskim."
                    }
                ]
            }
        ]

        # Dodaj obrazy do wiadomości
        for frame_path in frame_paths:
            with open(frame_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        # Wywołaj API OpenAI Vision
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )

        analysis = response.choices[0].message.content
        logger.info(f"Analiza przez OpenAI Vision zakończona sukcesem, długość tekstu: {len(analysis)}")

        # Sformatuj wynik analizy
        formatted_analysis = f"""
Analiza wizualna fragmentu wideo (ID: {segment_id}):

{analysis}

Informacje techniczne o wideo:
- Całkowity czas trwania: {duration:.2f} sekund
- Liczba analizowanych klatek: {len(frame_paths)} klatek
- Model analizy: OpenAI GPT-4o
        """

        return formatted_analysis

    except Exception as e:
        logger.error(f"Błąd podczas analizy zawartości wideo przez OpenAI Vision: {e}")
        import traceback
        logger.error(f"Stos błędów: {traceback.format_exc()}")
        return f"Błąd podczas analizy zawartości wideo: {str(e)}"


def generate_detailed_markdown_report(segment_info, transcript, visual_analysis, video_path, segment_duration=200, frames_per_segment=1):
    """
    Generuje szczegółowy raport w formacie Markdown z wszystkimi wynikami analizy wideo.

    Args:
        segment_info: Informacje o segmencie wideo
        transcript: Oryginalna transkrypcja audio
        visual_analysis: Analiza wizualna zawartości wideo
        video_path: Ścieżka do oryginalnego pliku wideo

    Returns:
        str: Zawartość raportu w formacie Markdown
    """
    import datetime

    # Pobierz informacje o pliku wideo
    video_name = os.path.basename(video_path)
    video_dir = os.path.dirname(video_path)
    segment_index = segment_info.get('index', 0)
    start_time = segment_info.get('start_time', 0)
    end_time = segment_info.get('end_time', 0)
    duration = segment_info.get('duration', 0)

    # Formatuj czasy
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    start_time_formatted = format_time(start_time)
    end_time_formatted = format_time(end_time)
    duration_formatted = format_time(duration)

    # Pobierz całkowity czas trwania wideo
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        total_duration = float(result.stdout.strip())
        total_duration_formatted = format_time(total_duration)
    except:
        total_duration_formatted = "Nieznany"

    # Przygotuj informacje o plikach
    mp4_path = segment_info.get('mp4_path', '')
    mp3_path = segment_info.get('mp3_path', '')

    # Generuj raport Markdown
    report = f"""# 📹 Raport Analizy Segmentu Wideo

## 📋 Informacje Ogólne

| Parametr | Wartość |
|----------|---------|
| **Nazwa pliku wideo** | `{video_name}` |
| **Ścieżka wideo** | `{video_path}` |
| **Data i czas analizy** | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
| **Całkowity czas trwania wideo** | {total_duration_formatted} |

## 🎬 Informacje o Segmencie

| Parametr | Wartość |
|----------|---------|
| **Numer segmentu** | {segment_index + 1} |
| **Czas rozpoczęcia** | {start_time_formatted} ({start_time:.2f}s) |
| **Czas zakończenia** | {end_time_formatted} ({end_time:.2f}s) |
| **Czas trwania segmentu** | {duration_formatted} ({duration:.2f}s) |
| **Pozycja w wideo** | {start_time_formatted} - {end_time_formatted} |

## 📁 Pliki Segmentu

### Plik Wideo
- **Ścieżka**: `{mp4_path}`
- **Rozmiar**: {os.path.getsize(mp4_path) if os.path.exists(mp4_path) else 'Nieznany'} bajtów
- **Format**: MP4 (segment wideo)

### Plik Audio
- **Ścieżka**: `{mp3_path}`
- **Rozmiar**: {os.path.getsize(mp3_path) if os.path.exists(mp3_path) else 'Nieznany'} bajtów
- **Format**: MP3 (wyodrębniony dźwięk)

---

## 🎵 Analiza Audio - Transkrypcja

### 📝 Transkrypcja Oryginalna
{transcript if transcript and transcript.strip() else "*Brak transkrypcji - sprawdź konfigurację API ElevenLabs*"}


### 📊 Statystyki Transkrypcji
| Metryka | Wartość |
|---------|---------|
| **Długość oryginalnego tekstu** | {len(transcript) if transcript else 0} znaków |
| **Model transkrypcji** | ElevenLabs Scribe V1 |

---

## 👁️ Analiza Wizualna Zawartości Wideo

### 📸 Opis Wizualny
{visual_analysis if visual_analysis and not visual_analysis.startswith("Nie") else "*Brak analizy wizualnej - sprawdź konfigurację API OpenAI*"}


### 🔍 Szczegóły Analizy Wizualnej
| Parametr | Wartość |
|----------|---------|
| **Model analizy** | OpenAI GPT-4o Vision |
| **Liczba klatek analizowanych** | {len([f for f in os.listdir(video_dir) if f.startswith(f"{os.path.splitext(video_name)[0]}_keyframe")]) if os.path.exists(video_dir) else 0} |
| **Język opisu** | Polski |
| **Głębokość analizy** | Szczegółowa (scena, postacie, działania, atmosfera) |

---

## 🏗️ Analiza Techniczna

### ⚙️ Parametry Przetwarzania
| Parametr | Wartość |
|----------|---------|
| **Długość segmentu** | {segment_duration}s (z konfiguracji) |
| **Liczba klatek na segment** | {frames_per_segment} (z konfiguracji) |
| **Format wyjściowy** | Markdown (.md) |
| **Kompatybilność z RAGFlow** | ✅ Tak |

### 🔧 Narzędzia Użyte
- **Segmentacja wideo**: FFmpeg
- **Wyodrębnianie klatek**: FFmpeg
- **Transkrypcja audio**: ElevenLabs Scribe V1 API
- **Analiza wizualna**: OpenAI GPT-4o Vision API
- **Generowanie raportu**: Python + Markdown

---

## 📈 Podsumowanie i Wnioski

### ✅ Zrealizowane Zadania
- [x] Segmentacja wideo na części
- [x] Wyodrębnienie ścieżki audio
- [x] Transkrypcja mowy na tekst
- [x] Analiza zawartości wizualnej
- [x] Generowanie szczegółowego raportu
- [x] Przygotowanie do przesłania do RAGFlow

### 📋 Jakość Wyników
| Komponent | Status | Jakość |
|-----------|--------|---------|
| **Transkrypcja** | {"✅ Dostępna" if transcript and transcript.strip() else "❌ Niedostępna"} | {"Wysoka" if transcript and len(transcript) > 50 else "Niska"} |
| **Analiza wizualna** | {"✅ Dostępna" if visual_analysis and not visual_analysis.startswith("Nie") else "❌ Niedostępna"} | {"Wysoka" if visual_analysis and len(visual_analysis) > 100 else "Niska"} |

### 🎯 Rekomendacje
1. **Dla lepszej transkrypcji**: Upewnij się, że jakość audio jest wysoka
2. **Dla dokładniejszej analizy wizualnej**: Zwiększ liczbę klatek kluczowych na segment
3. **Dla optymalizacji**: Dostosuj długość segmentów do zawartości wideo

---

## 🏷️ Tagi i Kategorie
**Tagi**: `wideo`, `analiza`, `transkrypcja`, `sztuczna inteligencja`, `RAGFlow`

**Kategorie**:
- 📹 Przetwarzanie wideo
- 🎵 Analiza audio
- 👁️ Analiza wizualna
- 📄 Generowanie raportów

---

*Raport wygenerowany automatycznie przez system VideoProcessor*  
*Data generowania: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    return report


def process_video_segment(segment_info, elevenlabs_api_key=None, openai_api_key=None, skip_video_analysis=False, frames_per_segment=1, segment_duration=200, video_path=None):
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

    Returns:
        Słownik z wynikami przetwarzania.
    """
    result = {
        'segment_info': segment_info,
        'transcript': None,
        'visual_analysis': None,
        'combined_result_path': None
    }

        # Transkrypcja audio na tekst
    if 'mp3_path' in segment_info:
        logger.info(f"Rozpoczynam przetwarzanie audio: {segment_info['mp3_path']}")
        transcript = transcribe_audio(segment_info['mp3_path'], elevenlabs_api_key)
        result['transcript'] = transcript    # Analiza zawartości wideo (jeśli nie pominięto)
    if 'mp4_path' in segment_info and not skip_video_analysis:
        logger.info(f"Rozpoczynam analizę zawartości wideo: {segment_info['mp4_path']}, wyodrębnianie {frames_per_segment} klatek")
        visual_analysis = analyze_video_content(segment_info['mp4_path'], openai_api_key, frames_per_segment)
        result['visual_analysis'] = visual_analysis
    elif skip_video_analysis:
        logger.info("Pominięto analizę zawartości wideo")
        result['visual_analysis'] = "Pominięto analizę zawartości wideo (konto API może mieć zaległości w płatnościach lub problemy z uprawnieniami)"

    # Połącz wyniki i zapisz do pliku Markdown
    output_dir = os.path.dirname(segment_info.get('mp4_path', ''))
    if output_dir:
        segment_name = os.path.splitext(os.path.basename(segment_info.get('mp4_path', '')))[0]
        result_path = os.path.join(output_dir, f"{segment_name}_analysis.md")

        # Wygeneruj szczegółowy raport Markdown
        markdown_content = generate_detailed_markdown_report(
            segment_info,
            result.get('transcript'),
            result.get('visual_analysis'),
            video_path or segment_info.get('mp4_path', ''),
            segment_duration,
            frames_per_segment
        )

        # Utwórz plik wynikowy
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        result['combined_result_path'] = result_path
        logger.info(f"Wyniki przetwarzania zostały zapisane do: {result_path}")

        # Zapisz również wyniki w formacie JSON
        json_result_path = os.path.join(output_dir, f"{segment_name}_result.json")
        with open(json_result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        result['json_result_path'] = json_result_path
        logger.info(f"Wyniki JSON zostały zapisane do: {json_result_path}")

    return result


def main():
    # Zadeklaruj zmienne globalne
    global RAGFLOW_API_URL, RAGFLOW_API_KEY
    
    # Załaduj konfigurację zmiennych środowiskowych
    load_environment_config()
    
    # Umieść ścieżki do swoich plików wideo na tej liście
    video_paths = [
        '/home/zy/ragflow/1101732139-1-208.mp4'
    ]
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Konfiguracja ścieżek wideo zakończona: {len(video_paths)} plików wideo")

    # Załaduj parametry konfiguracyjne ze zmiennych środowiskowych
    output_dir = os.getenv('OUTPUT_DIR', './video_segments')
    segment_duration = int(os.getenv('SEGMENT_DURATION', '200'))  # Jeden segment co 200 sekund
    dataset_name = os.getenv('DATASET_NAME', 'Video Segments Dataset')  # Domyślna nazwa zbioru danych
    frames_per_segment = int(os.getenv('FRAMES_PER_SEGMENT', '1'))  # Liczba klatek kluczowych do wyodrębnienia na segment

    # Konfiguracja przełączników funkcji
    skip_video_analysis = os.getenv('SKIP_VIDEO_ANALYSIS', 'false').lower() == 'true'
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Konfiguracja funkcji: Analiza wideo {'pominięta' if skip_video_analysis else 'włączona'}, {frames_per_segment} klatek na segment")

    # Ustaw poziom logowania dla requests na WARNING, aby zredukować niepotrzebne komunikaty
    import logging
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Konfiguracja RAGFlow zakończona, adres usługi: {RAGFLOW_API_URL}")

    # Pobierz klucze API dużych modeli ze zmiennych środowiskowych
    elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Konfiguracja kluczy API dużych modeli zakończona")

    # Zachowaj funkcję parsowania argumentów wiersza poleceń, ale ustaw jako opcjonalną
    if len(sys.argv) > 1:  # Jeśli podano argumenty wiersza poleceń
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

        args = parser.parse_args()

        # Jeśli wiersz poleceń określa wideo, nadpisz domyślną listę wideo
        if args.video:
            video_paths = [args.video]

        # Zaktualizuj inne parametry
        if args.output_dir:
            output_dir = args.output_dir
        if args.segment_duration:
            segment_duration = args.segment_duration
        if args.frames_per_segment:
            frames_per_segment = args.frames_per_segment
        if args.dataset_name:
            dataset_name = args.dataset_name
        # Jeśli argumenty wiersza poleceń nadpisują zmienne środowiskowe, zaktualizuj zmienne globalne
        if args.ragflow_api_url or args.ragflow_api_key:
            if args.ragflow_api_url:
                RAGFLOW_API_URL = args.ragflow_api_url
            if args.ragflow_api_key:
                RAGFLOW_API_KEY = args.ragflow_api_key

    # Utwórz katalog wyjściowy
    os.makedirs(output_dir, exist_ok=True)

    # Zapisz czas rozpoczęcia
    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Rozpoczynam przetwarzanie wideo...")

    # Przetwórz pierwsze wideo z listy
    if not video_paths:
        logger.error("Nie podano ścieżki do wideo, program zostanie zakończony")
        return

    video_path = video_paths[0]  # Przetwarzaj pierwsze wideo

    # 1. Segmentacja wideo
    logger.info(f"Rozpoczynam przetwarzanie wideo: {video_path}")
    segments = segment_video(video_path, output_dir, segment_duration)

    if not segments:
        logger.error("Segmentacja wideo nie powiodła się, program zostanie zakończony")
        return

    # 2. Przetwarzanie każdego segmentu wideo
    results = []
    for segment in segments:
        logger.info(f"Rozpoczynam przetwarzanie segmentu {segment['index'] + 1}/{len(segments)}")
        result = process_video_segment(segment, elevenlabs_api_key, openai_api_key, skip_video_analysis, frames_per_segment, segment_duration=segment_duration, video_path=video_path)
        results.append(result)

    # 3. Przesyłanie do RAGFlow
    if RAGFLOW_API_KEY:
        logger.info(f"Rozpoczynam przesyłanie wyników do RAGFlow, nazwa zbioru danych: {dataset_name}")

        # Najpierw spróbuj pobrać istniejący zbiór danych
        dataset_id = get_dataset_by_name(dataset_name)

        # Jeśli zbiór danych nie istnieje, utwórz nowy
        if not dataset_id:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Zbiór danych nie istnieje, próba utworzenia nowego zbioru danych...")
            dataset_id = create_ragflow_dataset(dataset_name)
            if not dataset_id:
                logger.error("Nie udało się utworzyć zbioru danych RAGFlow, nie można kontynuować")
                return
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pomyślnie utworzono zbiór danych RAGFlow")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Znaleziono istniejący zbiór danych, ID: {dataset_id}, zostanie on użyty bezpośrednio")

        # Weryfikacja konfiguracji grafu wiedzy zbioru danych
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔍 Weryfikacja konfiguracji grafu wiedzy zbioru danych...")
        config_result = verify_dataset_graphrag_config(dataset_id)
        if config_result["success"]:
            if config_result["enabled"]:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Funkcja grafu wiedzy dla zbioru danych jest poprawnie włączona")
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Ostrzeżenie: Funkcja grafu wiedzy dla zbioru danych nie jest włączona, co może wpłynąć na działanie asystenta")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ Nie można zweryfikować konfiguracji grafu wiedzy zbioru danych: {config_result.get('error')}")

        # Przesyłaj tylko dokumenty z wynikami analizy, nie przesyłaj plików MP3 i MP4
        document_ids = []
        for result in results:
            if 'combined_result_path' in result and result['combined_result_path']:
                file_path = result['combined_result_path']
                file_name = os.path.basename(file_path)

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Przesyłanie dokumentu z wynikami analizy: {file_name}")
                # Podczas przesyłania ustaw auto_parse=False, aby później jednorazowo uruchomić parsowanie wszystkich dokumentów
                document_id = upload_file_to_ragflow(dataset_id, file_name, file_path, auto_parse=False)
                if document_id:
                    # Po pomyślnym przesłaniu, natychmiast zaktualizuj konfigurację parsera dokumentu
                    update_success = update_document_parser_config(dataset_id, document_id)
                    if update_success:
                        document_ids.append(document_id)
                        print(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dokument {file_name} (ID: {document_id}) został dodany do kolejki parsowania")
                    else:
                        print(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ostrzeżenie: Aktualizacja konfiguracji parsowania dla dokumentu {file_name} (ID: {document_id}) nie powiodła się")

        # Uruchom parsowanie wszystkich dokumentów wsadowo
        if document_ids:
            logger.info(f"Rozpoczynam wsadowe uruchamianie parsowania wszystkich dokumentów, liczba dokumentów: {len(document_ids)}")
            # Dla każdego dokumentu uruchom parsowanie osobno, aby uniknąć problemów, które mogą wystąpić przy przetwarzaniu wsadowym
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
                # Dodaj krótkie opóźnienie, aby uniknąć zbyt częstych żądań
                time.sleep(0.5)

            logger.info(f"Uruchomiono parsowanie dla {success_count}/{len(document_ids)} dokumentów")

            # Oczekiwanie na zakończenie parsowania dokumentów
            if triggered_doc_ids:
                logger.info("Rozpoczynam oczekiwanie na zakończenie parsowania dokumentów...")
                parsing_result = wait_for_documents_parsing(
                    dataset_id,
                    triggered_doc_ids,
                    check_interval=30  # Sprawdzaj co 30 sekund
                )

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Statystyki zakończenia parsowania dokumentów:")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - Całkowita liczba dokumentów: {parsing_result['total']}")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - Sparsowano pomyślnie: {parsing_result['completed']}")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - Błędy parsowania: {parsing_result['failed']}")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - Czas parsowania: {parsing_result['elapsed_time']:.1f} sekund")

                # Po zakończeniu parsowania dokumentów, poinformuj użytkownika, jak utworzyć asystenta
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

        # Zapisz czas zakończenia i całkowity czas trwania
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