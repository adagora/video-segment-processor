# VideoProcessor
processing video and upload this to RAGFlow

## FEATURES
- llm vision to detect action on image
- audio *.mp3 -> transcript scribe_v1, 11labs -> text
- resports uploaded to RAG DB (RAGFLOW)
- timestamp
- analyse more frames per segment

## .env
```
RAGFLOW_API_URL=http://
RAGFLOW_API_KEY=REDACTED
OPENAI_API_KEY=REDACTED

ELEVENLABS_API_KEY=REDACTED

SEGMENT_DURATION=200
OUTPUT_DIR=./video_segments
DATASET_NAME=zbior_danych_wideo
SKIP_VIDEO_ANALYSIS=false
FRAMES_PER_SEGMENT=5
```

### install 
```
pip install -r requirements.txt
sudo apt install ffmpeg
choco install ffmpeg or winget install "FFmpeg (Essentials Build)" (windows)
```

### run
```
python video_segment_processor.py --video "Nowości 2025- Modest Ligfht.mp4" --segment-duration 60
```

## Other
```
ffprobe -v quiet -print_format json -show_format -show_streams "Nowości 2025- Modest Ligfht_segment_0.mp3"
```

## Timestamp
```
python video_segment_processor.py --video "video_input\Szablon prezentacji Bezpieczeństwo 1 — no password manager.mp4" --timestamps --segment-duration 60
```

