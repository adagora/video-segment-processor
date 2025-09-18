#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Report Generator Module

This module provides functionality for generating comprehensive reports from video analysis results.
It combines transcripts, visual analysis, and metadata from multiple video segments into well-formatted
Markdown reports suitable for knowledge base ingestion and AI assistant interactions.
"""

import os
import datetime
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VideoReportGenerator:
    """
    A class for generating comprehensive video analysis reports.
    
    This class handles the creation of comprehensive reports that combine 
    all segments from a video processing session into a single document.
    """
    
    def __init__(self):
        """Initialize the report generator."""
        pass
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Format time in seconds to HH:MM:SS or MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def get_video_duration(video_path: str) -> tuple[float, str]:
        """
        Get the total duration of a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (duration_in_seconds, formatted_duration_string)
        """
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
        """
        Find the keyframe image for a given segment.
        
        Args:
            segment_info: Segment information dictionary
            video_name_clean: Clean video name without extension
            
        Returns:
            Path to keyframe image if found, None otherwise
        """
        segment_index = segment_info.get('index', 0)
        output_dir = os.path.dirname(segment_info.get('mp4_path', ''))
        keyframe_path = os.path.join(output_dir, f"{video_name_clean}_segment_{segment_index}_keyframe.jpg")
        
        if os.path.exists(keyframe_path):
            return keyframe_path
        return None
    
    def generate_comprehensive_report(self, all_results: List[Dict[str, Any]], 
                                    video_path: str, segment_duration: int = 200, 
                                    frames_per_segment: int = 1) -> str:
        """
        Generate a comprehensive report combining all video segments.
        
        This method creates a single comprehensive report that includes:
        - Combined transcripts from all segments with embedded keyframe images
        - Consolidated visual analysis from all segments
        - Complete metadata and statistics
        
        Args:
            all_results: List of processing results from all segments
            video_path: Path to the original video file
            segment_duration: Duration of segments in seconds
            frames_per_segment: Number of frames analyzed per segment
            
        Returns:
            Comprehensive Markdown report content as string
        """
        video_name = os.path.basename(video_path)        
        _, total_duration_formatted = self.get_video_duration(video_path)
        
        # Combine all transcripts with embedded images
        combined_transcript = ""
        visual_descriptions = []
        
        for result in all_results:
            segment_info = result.get('segment_info', {})
            segment_index = segment_info.get('index', 0)
            
            # Use granular transcript if available, fallback to original
            granular_transcript = result.get('granular_transcript', '')
            original_transcript = result.get('transcript', '')
            
            if granular_transcript and granular_transcript.strip() and granular_transcript != original_transcript:
                combined_transcript += granular_transcript + "\n\n"
            elif original_transcript and original_transcript.strip():
                combined_transcript += original_transcript + "\n\n"
            
            # Collect visual analyses for separate section
            visual_analysis = result.get('visual_analysis', '')
            if (visual_analysis and visual_analysis.strip() and 
                not visual_analysis.startswith("Nie") and 
                not visual_analysis.startswith("Pominięto")):
                
                visual_descriptions.append(visual_analysis)
        
        # Combine visual descriptions with separators
        combined_visual_analysis = "\n\n---\n\n".join(visual_descriptions).strip()
        if not combined_visual_analysis:
            combined_visual_analysis = "Brak analizy wizualnej - sprawdź konfigurację API OpenAI lub została pominięta"
        
        # Generate comprehensive report in the requested format
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# Raport - {video_name}

## Transcript

{combined_transcript.strip()}

### Details
- nazwa pliku: {video_name}
- ścieżka wideo: {video_path}
- data i czas analizy: {current_time}
- całkowity czas trwania: {total_duration_formatted}
- model transkrypcyjny: ElevenLabs Scribe V1

## Analiza Wizualna
{combined_visual_analysis}

### Details
- Głębokość analizy: Szczegółowa (scena, postacie, działania, atmosfera)
- model analizy: OpenAI GPT-4o Vision

## Data generowania
{current_time}
"""
        return report
    
    def save_report(self, report_content: str, output_path: str) -> bool:
        """
        Save report content to a file.
        
        Args:
            report_content: The report content to save
            output_path: Path where to save the report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
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
                                    frames_per_segment: int = 1) -> Optional[str]:
    """
    Convenience function to create and save a comprehensive video report.
    
    Args:
        all_results: List of processing results from all segments
        video_path: Path to the original video file
        output_dir: Directory where to save the report
        segment_duration: Duration of segments in seconds
        frames_per_segment: Number of frames analyzed per segment
        
    Returns:
        Path to the saved report file, or None if failed
    """
    generator = VideoReportGenerator()
    
    # Generate the comprehensive report
    report_content = generator.generate_comprehensive_report(
        all_results, video_path, segment_duration, frames_per_segment
    )
    
    # Determine output path
    video_name_clean = os.path.splitext(os.path.basename(video_path))[0]
    report_path = os.path.join(output_dir, f"{video_name_clean}_comprehensive_report.md")
    
    # Save the report
    if generator.save_report(report_content, report_path):
        return report_path
    return None


