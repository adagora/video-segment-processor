#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import datetime
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VideoReportGenerator:
    def __init__(self):
        pass
    
    @staticmethod
    def format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def get_video_duration(video_path: str) -> tuple[float, str]:
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            duration = float(result.stdout.strip())
            formatted = VideoReportGenerator.format_time(duration)
            return duration, formatted
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
            return 0.0, "Unknown"
    
    @staticmethod
    def find_keyframe_image(segment_info: Dict[str, Any], video_name_clean: str) -> Optional[str]:
        segment_index = segment_info.get('index', 0)
        output_dir = os.path.dirname(segment_info.get('mp4_path', ''))
        keyframe_path = os.path.join(output_dir, f"{video_name_clean}_segment_{segment_index}_keyframe.jpg")
        
        if os.path.exists(keyframe_path):
            return keyframe_path
        return None
    
    def generate_comprehensive_report(self, all_results: List[Dict[str, Any]], 
                                    video_path: str, segment_duration: int = 200, 
                                    global_visual_analysis: Optional[str] = None) -> str:
        video_name = os.path.basename(video_path)        
        _, total_duration_formatted = self.get_video_duration(video_path)
        
        combined_transcript = ""
        
        for result in all_results:
            segment_info = result.get('segment_info', {})
            segment_index = segment_info.get('index', 0)
            
            granular_transcript = result.get('granular_transcript', '')
            original_transcript = result.get('transcript', '')
            
            if granular_transcript and granular_transcript.strip() and granular_transcript != original_transcript:
                combined_transcript += granular_transcript + "\n\n"
            elif original_transcript and original_transcript.strip():
                combined_transcript += original_transcript + "\n\n"
        
        if global_visual_analysis:
            combined_visual_analysis = global_visual_analysis
        else:
            combined_visual_analysis = "Brak analizy wizualnej - sprawdź konfigurację API OpenAI lub została pominięta"
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# Raport - {video_name}

## Transcript

{combined_transcript.strip()}

### Detale
- nazwa pliku: {video_name}
- ścieżka wideo: {video_path}
- data i czas analizy: {current_time}
- całkowity czas trwania: {total_duration_formatted}
- model transkrypcyjny: ElevenLabs Scribe V1

## Analiza Wizualna
{combined_visual_analysis}

### Detale
- Głębokość analizy: Szczegółowa (scena, postacie, działania, atmosfera)
- model analizy: OpenAI GPT-4o Vision

## Data generowania
{current_time}
"""
        return report
    
    def save_report(self, report_content: str, output_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Report saved successfully to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save report to {output_path}: {e}")
            return False


def create_comprehensive_video_report(all_results: List[Dict[str, Any]], 
                                    video_path: str, output_dir: str,
                                    segment_duration: int = 200, 
                                    global_visual_analysis: Optional[str] = None) -> Optional[str]:
    generator = VideoReportGenerator()
    
    report_content = generator.generate_comprehensive_report(
        all_results, video_path, segment_duration, global_visual_analysis=global_visual_analysis
    )
    
    video_name_clean = os.path.splitext(os.path.basename(video_path))[0]
    report_path = os.path.join(output_dir, f"{video_name_clean}_comprehensive_report.md")
    
    if generator.save_report(report_content, report_path):
        return report_path
    return None
