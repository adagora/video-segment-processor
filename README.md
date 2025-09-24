# VideoProcessor
processing video and upload this to RAGFlow

## FEATURES
- llm vision to detect action on image
- audio *.mp3 -> transcript scribe_v1, 11labs -> text
- resports markdown uploaded to RAG DB (RAGFLOW)
- timestamp overlay on image
- automatic scene /frame detection
- manual scene /frame detection

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
```

### install 
```
pip install -r requirements.txt
sudo apt install ffmpeg
choco install ffmpeg or winget install "FFmpeg (Essentials Build)" (windows)
```

### run
```
automatic:
python video_segment_processor.py --video "video_input/Nowości 2025- Modest Ligfht.mp4" --segment-duration 60 --enable-video-analysis --timestamps

python video_segment_processor.py --video "video_input/Szablon prezentacji Bezpieczeństwo 1 — no password manager.mp4" --segment-duration 60 --enable-video-analysis --timestamps

manual timestamp:
python video_segment_processor.py --video "video_input/Nowości 2025- Modest Ligfht.mp4" --segment-duration 60 --manual-scenes "0:02,0:03,0:06,0:12,0:35,2:00,2:33,3:05,4:13,4:25,5:25,5:52,6:00,6:30" --enable-video-analysis --timestamps
```

## Timestamp
```
python video_segment_processor.py --video "video_input/Nowości 2025- Modest Ligfht.mp4" --segment-duration 60 --enable-video-analysis --timestamps
```


## testing
```
scene detection:
python test_scene_detection.py
```