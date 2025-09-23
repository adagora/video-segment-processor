#!/usr/bin/env python3

import os
import subprocess
import re
import json
import time
from datetime import datetime
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scene_detection_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SceneDetectionTester:
    def __init__(self):
        self.test_results = []
        
    def get_video_info(self, video_path):
        """Get basic video information using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            duration = float(info['format']['duration'])
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            fps = eval(video_stream['r_frame_rate'])  # Convert fraction to float
            
            return {
                'duration': duration,
                'width': width,
                'height': height,
                'fps': fps,
                'format': info['format']['format_name'],
                'size_mb': float(info['format']['size']) / (1024*1024)
            }
        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {e}")
            return None
    
    def detect_scenes(self, video_path, threshold=0.02):
        """
        Detect scene changes using FFmpeg scene detection
        
        Args:
            video_path: Path to video file
            threshold: Scene detection threshold (0.01-0.05 for presentations, 0.1-0.3 for action videos)
            
        Returns:
            Dictionary with scene detection results
        """
        logger.info(f"Detecting scenes in {os.path.basename(video_path)} with threshold {threshold}")
        
        start_time = time.time()
        
        try:
            # Run FFmpeg scene detection
            scene_cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-filter_complex', f"select='bitor(gt(scene\\,{threshold})\\,eq(n\\,0))',showinfo",
                '-f', 'null', '-'
            ]
            
            scene_result = subprocess.run(
                scene_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True
            )
            
            processing_time = time.time() - start_time
            
            if scene_result.returncode != 0:
                logger.error(f"FFmpeg scene detection failed: {scene_result.stdout}")
                return None
            
            # Parse scene detection output
            scene_times = []
            scene_scores = []
            
            for line in scene_result.stdout.splitlines():
                if 'n:' in line and 'pts_time:' in line:
                    # Extract timestamp
                    pts_match = re.search(r'pts_time:([\d.]+)', line)
                    if pts_match:
                        pts_time = float(pts_match.group(1))
                        scene_times.append(pts_time)
                        
                        # Try to extract scene score if available
                        score_match = re.search(r'scene:([\d.]+)', line)
                        if score_match:
                            scene_scores.append(float(score_match.group(1)))
                        else:
                            scene_scores.append(threshold)  # Default to threshold if not found
            
            # Remove duplicates and sort
            scene_data = list(zip(scene_times, scene_scores))
            scene_data = sorted(set(scene_data))
            scene_times, scene_scores = zip(*scene_data) if scene_data else ([], [])
            
            result = {
                'threshold': threshold,
                'scene_count': len(scene_times),
                'scene_times': list(scene_times),
                'scene_scores': list(scene_scores),
                'processing_time': processing_time,
                'avg_interval': (max(scene_times) - min(scene_times)) / (len(scene_times) - 1) if len(scene_times) > 1 else 0
            }
            
            logger.info(f"Found {len(scene_times)} scene changes in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in scene detection: {e}")
            return None
    
    def extract_keyframes_at_scenes(self, video_path, scene_times, output_dir, max_frames=10, add_timestamps=False):
        """Extract keyframes at detected scene change times"""
        if not scene_times:
            logger.warning("No scene times provided for keyframe extraction")
            return []
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        
        extracted_frames = []
        
        # Limit to max_frames evenly distributed
        if len(scene_times) > max_frames:
            step = len(scene_times) / max_frames
            selected_times = [scene_times[int(i * step)] for i in range(max_frames)]
        else:
            selected_times = scene_times
        
        logger.info(f"Extracting {len(selected_times)} keyframes to: {output_dir}")
        
        for i, scene_time in enumerate(selected_times):
            frame_path = os.path.join(output_dir, f"{video_name}_scene_{i+1:02d}_t{scene_time:.2f}.jpg")
            
            if add_timestamps:
                # Add timestamp overlay to the image for visual inspection
                minutes = int(scene_time // 60)
                seconds = int(scene_time % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                
                extract_cmd = [
                    'ffmpeg', '-y', '-ss', str(scene_time),
                    '-i', video_path, '-vframes', '1',
                    '-vf', f"drawtext=fontcolor=yellow:fontsize=24:box=1:boxcolor=black@0.8:boxborderw=2:text='{time_str}':x=10:y=10",
                    '-q:v', '2', frame_path
                ]
            else:
                extract_cmd = [
                    'ffmpeg', '-y', '-ss', str(scene_time),
                    '-i', video_path, '-vframes', '1',
                    '-q:v', '2', frame_path
                ]
            
            try:
                subprocess.run(extract_cmd, check=True, capture_output=True)
                extracted_frames.append(frame_path)
                logger.info(f"Keyframe {i+1:2d}/{len(selected_times)} at {scene_time:6.2f}s -> {os.path.basename(frame_path)}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract keyframe at {scene_time:.2f}s: {e}")
        
        logger.info(f"Successfully extracted {len(extracted_frames)} keyframes")
        return extracted_frames
    
    def test_multiple_thresholds(self, video_path, thresholds=None):
        """Test scene detection with multiple threshold values"""
        if thresholds is None:
            # Default thresholds: very sensitive to very loose
            thresholds = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]
        
        video_info = self.get_video_info(video_path)
        if not video_info:
            logger.error(f"Cannot get video info for {video_path}")
            return None
        
        logger.info(f"\nTesting video: {os.path.basename(video_path)}")
        logger.info(f"Duration: {video_info['duration']:.2f}s, Resolution: {video_info['width']}x{video_info['height']}")
        logger.info(f"FPS: {video_info['fps']:.2f}, Size: {video_info['size_mb']:.1f}MB")
        
        test_result = {
            'video_path': video_path,
            'video_info': video_info,
            'threshold_tests': [],
            'test_timestamp': datetime.now().isoformat()
        }
        
        for threshold in thresholds:
            scene_result = self.detect_scenes(video_path, threshold)
            if scene_result:
                test_result['threshold_tests'].append(scene_result)
                
                # Print summary for this threshold
                logger.info(f"Threshold {threshold}: {scene_result['scene_count']} scenes, "
                          f"avg interval: {scene_result['avg_interval']:.2f}s")
        
        self.test_results.append(test_result)
        return test_result
    
    def analyze_results(self, video_result):
        """Analyze results and provide recommendations"""
        if not video_result or not video_result['threshold_tests']:
            return None
        
        video_name = os.path.basename(video_result['video_path'])
        duration = video_result['video_info']['duration']
        
        logger.info(f"\n=== ANALYSIS FOR {video_name} ===")
        
        # Find optimal thresholds
        threshold_analysis = []
        for test in video_result['threshold_tests']:
            scenes_per_minute = (test['scene_count'] / duration) * 60
            threshold_analysis.append({
                'threshold': test['threshold'],
                'scene_count': test['scene_count'],
                'scenes_per_minute': scenes_per_minute,
                'avg_interval': test['avg_interval']
            })
        
        # Sort by scenes per minute for analysis
        threshold_analysis.sort(key=lambda x: x['scenes_per_minute'])
        
        logger.info("Threshold Analysis:")
        logger.info("Threshold | Scenes | Scenes/min | Avg Interval")
        logger.info("-" * 50)
        for analysis in threshold_analysis:
            logger.info(f"{analysis['threshold']:8.2f} | {analysis['scene_count']:6d} | "
                       f"{analysis['scenes_per_minute']:9.2f} | {analysis['avg_interval']:11.2f}s")
        
        # Recommendations based on video type
        if "prezentacji" in video_name.lower() or "szablon" in video_name.lower():
            # Presentation video - recommend 1-3 scenes per minute
            recommended = [a for a in threshold_analysis if 1 <= a['scenes_per_minute'] <= 3]
            video_type = "Presentation"
        else:
            # General video - recommend 0.5-2 scenes per minute
            recommended = [a for a in threshold_analysis if 0.5 <= a['scenes_per_minute'] <= 2]
            video_type = "General Video"
        
        logger.info(f"\nVideo Type: {video_type}")
        if recommended:
            best = recommended[len(recommended)//2]  # Pick middle recommendation
            logger.info(f"RECOMMENDED THRESHOLD: {best['threshold']}")
            logger.info(f"This will detect ~{best['scenes_per_minute']:.1f} scenes per minute")
            logger.info(f"Expected {best['scene_count']} scenes total")
        else:
            logger.info("No threshold fits the recommended range. Consider custom values.")
        
        return {
            'video_type': video_type,
            'recommended_threshold': recommended[len(recommended)//2]['threshold'] if recommended else None,
            'analysis': threshold_analysis
        }
    
    def save_results(self, output_file="scene_detection_results.json"):
        """Save all test results to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Test scene detection on videos')
    parser.add_argument('--video', help='Specific video to test')
    parser.add_argument('--thresholds', nargs='+', type=float, 
                       help='Custom thresholds to test (e.g., --thresholds 0.01 0.02 0.05)')
    parser.add_argument('--extract-keyframes', action='store_true',
                       help='Extract keyframes at detected scene changes')
    parser.add_argument('--add-timestamps', action='store_true',
                       help='Add timestamp overlays to extracted keyframes for easier inspection')
    parser.add_argument('--max-keyframes', type=int, default=15,
                       help='Maximum number of keyframes to extract per video (default: 15)')
    parser.add_argument('--output-dir', default='scene_test_output',
                       help='Directory for output files')
    
    args = parser.parse_args()
    
    tester = SceneDetectionTester()
    
    # Default videos to test
    default_videos = [
        r"video_input\Nowości 2025- Modest Ligfht.mp4",
        r"video_input\Szablon prezentacji Bezpieczeństwo 1 — no password manager.mp4"
    ]
    
    videos_to_test = [args.video] if args.video else default_videos
    
    # Test each video
    for video_path in videos_to_test:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            continue
        
        # Test with multiple thresholds
        result = tester.test_multiple_thresholds(video_path, args.thresholds)
        
        if result:
            # Analyze and provide recommendations
            analysis = tester.analyze_results(result)
            
            # Extract keyframes if requested
            if args.extract_keyframes and analysis and analysis['recommended_threshold']:
                threshold = analysis['recommended_threshold']
                scene_detection = tester.detect_scenes(video_path, threshold)
                if scene_detection and scene_detection['scene_times']:
                    keyframe_dir = os.path.join(args.output_dir, 
                                              os.path.splitext(os.path.basename(video_path))[0])
                    frames = tester.extract_keyframes_at_scenes(
                        video_path, scene_detection['scene_times'], keyframe_dir,
                        max_frames=args.max_keyframes, add_timestamps=args.add_timestamps
                    )
                    logger.info(f"Extracted {len(frames)} keyframes to {keyframe_dir}")
                    logger.info(f"You can view the keyframes in: {os.path.abspath(keyframe_dir)}")
    
    # Save all results
    results_file = os.path.join(args.output_dir, "scene_detection_results.json")
    tester.save_results(results_file)
    
    logger.info(f"\n=== SCENE DETECTION TEST COMPLETED ===")
    logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
    logger.info(f"Results file: {os.path.abspath(results_file)}")
    if args.extract_keyframes:
        logger.info(f"View extracted keyframes in the subdirectories above")

if __name__ == "__main__":
    main()