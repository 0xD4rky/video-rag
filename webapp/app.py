"""
Flask web application for Video RAG Search
Inspired by Twelve Labs' minimalist design
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import datetime

sys.path.append('/Users/darky/Documents/video-rag/src')
from agent import VideoSearchAgent
from scene import get_video_info

app = Flask(__name__)
app.config['SECRET_KEY'] = 'video-rag-secret-key'
app.config['UPLOAD_FOLDER'] = '/Users/darky/Documents/video-rag/data/uploads'
app.config['OUTPUT_FOLDER'] = '/Users/darky/Documents/video-rag/data/output'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  

socketio = SocketIO(app, cors_allowed_origins="*")

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

# Initialize the video search agent
video_agent = VideoSearchAgent()

# Store processing logs
processing_logs = []

class WebSocketHandler(logging.Handler):
    """Custom logging handler to emit logs via WebSocket"""
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'level': record.levelname,
            'message': self.format(record),
            'module': record.name
        }
        processing_logs.append(log_entry)
        socketio.emit('log_update', log_entry)

# Set up logging
logging.basicConfig(level=logging.INFO)
websocket_handler = WebSocketHandler()
websocket_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(websocket_handler)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid conflicts
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Get video metadata
            video_info = get_video_info(file_path)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': file_path,
                'metadata': {
                    'duration': f"{video_info['duration']:.1f}s",
                    'fps': f"{video_info['fps']:.1f}",
                    'resolution': f"{int(video_info['width'])}x{int(video_info['height'])}",
                    'frames': int(video_info['frame_count'])
                }
            })
        except Exception as e:
            return jsonify({'error': f'Failed to process video: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/search', methods=['POST'])
def search_video():
    """Handle video search request"""
    global processing_logs
    processing_logs = []  # Clear previous logs
    
    data = request.get_json()
    video_path = data.get('video_path')
    query = data.get('query')
    top_n = data.get('top_n', 3)
    
    if not video_path or not query:
        return jsonify({'error': 'Video path and query are required'}), 400
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    try:
        # Clear previous output files
        output_dir = app.config['OUTPUT_FOLDER']
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # Run the video search
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Emit start processing event
        socketio.emit('processing_start', {'query': query, 'video': os.path.basename(video_path)})
        
        clips = loop.run_until_complete(video_agent.run(video_path, query, int(top_n)))
        loop.close()
        
        # Get output video files with metadata
        output_videos = []
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    file_path = os.path.join(output_dir, file)
                    try:
                        video_info = get_video_info(file_path)
                        # Extract score from filename if available
                        score = None
                        if '_score_' in file:
                            try:
                                score = float(file.split('_score_')[1].split('.')[0])
                            except:
                                pass
                        
                        output_videos.append({
                            'filename': file,
                            'path': file_path,
                            'metadata': {
                                'duration': f"{video_info['duration']:.1f}s",
                                'fps': f"{video_info['fps']:.1f}",
                                'resolution': f"{int(video_info['width'])}x{int(video_info['height'])}",
                                'frames': int(video_info['frame_count']),
                                'score': f"{score:.3f}" if score else "N/A"
                            }
                        })
                    except Exception as e:
                        logging.error(f"Error getting metadata for {file}: {e}")
        
        # Sort by score if available
        output_videos.sort(key=lambda x: float(x['metadata']['score']) if x['metadata']['score'] != "N/A" else 0, reverse=True)
        
        socketio.emit('processing_complete', {
            'clips_found': len(output_videos),
            'query': query
        })
        
        return jsonify({
            'success': True,
            'clips_found': len(output_videos),
            'output_videos': output_videos,
            'logs': processing_logs
        })
    
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        logging.error(error_msg)
        socketio.emit('processing_error', {'error': error_msg})
        return jsonify({'error': error_msg}), 500

@app.route('/api/video/<filename>')
def serve_video(filename):
    """Serve video files"""
    try:
        # Check both upload and output directories
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if os.path.exists(output_path):
            return send_file(output_path)
        elif os.path.exists(upload_path):
            return send_file(upload_path)
        else:
            return jsonify({'error': 'Video not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs')
def get_logs():
    """Get current processing logs"""
    return jsonify({'logs': processing_logs})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to Video RAG server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
