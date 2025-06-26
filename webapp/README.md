# Video RAG Web Interface

A modern, minimalist web interface for intelligent video search powered by AI. Inspired by Twelve Labs' design philosophy.

## Features

### 🎬 **Video Upload & Processing**
- Drag and drop video uploads
- Support for multiple formats (MP4, AVI, MOV, MKV, WEBM)
- Real-time video metadata display
- File validation and size limits

### 🔍 **Intelligent Search**
- Natural language queries
- Configurable number of result clips
- Real-time processing status
- Contextual search results

### 📊 **Live Processing Logs**
- Real-time log streaming via WebSocket
- Clean, readable log formatting
- Color-coded log levels (INFO, WARNING, ERROR)
- Scrollable log history

### 🎥 **Result Display**
- Grid layout for video clips
- Hover effects with metadata preview
- Relevance score display
- Full-screen video modal

### 🎨 **Modern UI/UX**
- Twelve Labs-inspired minimalist design
- Responsive layout for all devices
- Smooth animations and transitions
- Toast notifications for user feedback

## Setup Instructions

### Prerequisites
- Python 3.8+
- All dependencies from the main video-rag project

### Quick Start

1. **Navigate to the webapp directory:**
   ```bash
   cd /Users/darky/Documents/video-rag/webapp
   ```

2. **Run the startup script:**
   ```bash
   ./start.sh
   ```

3. **Open your browser and visit:**
   ```
   http://localhost:5000
   ```

### Manual Setup

If you prefer manual setup:

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r ../requirements.txt
   ```

3. **Start the application:**
   ```bash
   python app.py
   ```

## Usage Guide

### 1. Upload Video
- Click the upload area or drag and drop a video file
- Supported formats: MP4, AVI, MOV, MKV, WEBM
- Maximum file size: 500MB
- View automatic metadata extraction

### 2. Search Content
- Enter a natural language query describing what you're looking for
- Examples:
  - "person walking in the park"
  - "car chase scene"
  - "people talking at a table"
- Select the number of clips to extract (1-5)
- Click "Search Video"

### 3. Monitor Processing
- Watch real-time logs during processing
- See step-by-step progress updates
- Processing status indicators

### 4. View Results
- Browse results in a clean grid layout
- Hover over videos to see metadata
- Click videos to view in full-screen modal
- Relevance scores for each clip

## Technical Architecture

### Backend (Flask + SocketIO)
- **Flask**: Web framework and API endpoints
- **SocketIO**: Real-time log streaming
- **Integration**: Direct integration with existing video-rag agents

### Frontend (Vanilla JS + Modern CSS)
- **Real-time UI**: WebSocket-based live updates
- **Responsive Design**: Mobile-first approach
- **Modern CSS**: CSS Grid, Flexbox, CSS Variables
- **Accessibility**: Keyboard navigation, ARIA labels

### File Structure
```
webapp/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── start.sh           # Startup script
├── README.md          # This file
├── static/
│   ├── style.css      # Modern CSS styling
│   └── script.js      # Interactive JavaScript
└── templates/
    └── index.html     # Main HTML template
```

## API Endpoints

### Upload Video
```
POST /api/upload
Content-Type: multipart/form-data
Body: video file

Response: {
    "success": true,
    "filename": "video.mp4",
    "filepath": "/path/to/video.mp4",
    "metadata": {
        "duration": "120.5s",
        "fps": "30.0",
        "resolution": "1920x1080",
        "frames": 3615
    }
}
```

### Search Video
```
POST /api/search
Content-Type: application/json
Body: {
    "video_path": "/path/to/video.mp4",
    "query": "search query",
    "top_n": 3
}

Response: {
    "success": true,
    "clips_found": 3,
    "output_videos": [...],
    "logs": [...]
}
```

### Serve Video
```
GET /api/video/<filename>
Response: Video file stream
```

## Design Philosophy

This interface is inspired by Twelve Labs' approach to video AI:

- **Minimalist**: Clean, uncluttered interface focusing on core functionality
- **Intelligent**: AI-powered search with natural language understanding
- **Fast**: Real-time feedback and processing status
- **Accessible**: Intuitive design that's easy to use for everyone

## Customization

### Styling
Edit `static/style.css` to customize:
- Color scheme (CSS variables in `:root`)
- Layout and spacing
- Typography and fonts
- Animation timing

### Functionality
Edit `static/script.js` to modify:
- Upload handling
- Search behavior
- UI interactions
- WebSocket events

### Backend
Edit `app.py` to customize:
- API endpoints
- File handling
- Processing logic
- Error handling

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Port conflicts**: Change port in `app.py` if 5000 is occupied
3. **File upload failures**: Check file size and format restrictions
4. **Processing errors**: Verify main video-rag project setup

### Debug Mode
To enable debug mode, set `debug=True` in the `socketio.run()` call in `app.py`.

## Contributing

1. Follow the existing code style
2. Test all functionality before submitting
3. Update documentation for any changes
4. Ensure responsive design principles

## License

This project is part of the Video RAG system and follows the same licensing terms.
