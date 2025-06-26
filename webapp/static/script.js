// Video RAG Web Interface JavaScript
class VideoRAGApp {
    constructor() {
        this.socket = null;
        this.currentVideo = null;
        this.isProcessing = false;
        
        this.initializeElements();
        this.bindEvents();
        this.initializeSocket();
    }
    
    initializeElements() {
        // Upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.videoInput = document.getElementById('videoInput');
        this.videoPreview = document.getElementById('videoPreview');
        this.changeVideoBtn = document.getElementById('changeVideo');
        
        // Video metadata elements
        this.videoName = document.getElementById('videoName');
        this.videoDuration = document.getElementById('videoDuration');
        this.videoResolution = document.getElementById('videoResolution');
        this.videoFPS = document.getElementById('videoFPS');
        this.videoFrames = document.getElementById('videoFrames');
        
        // Search elements
        this.searchQuery = document.getElementById('searchQuery');
        this.topNSelect = document.getElementById('topN');
        this.searchBtn = document.getElementById('searchBtn');
        
        // Processing elements
        this.processingSection = document.getElementById('processingSection');
        this.processingText = document.getElementById('processingText');
        this.logsContent = document.getElementById('logsContent');
        this.clearLogsBtn = document.getElementById('clearLogs');
        
        // Results elements
        this.resultsSection = document.getElementById('resultsSection');
        this.resultsSummary = document.getElementById('resultsSummary');
        this.videoGrid = document.getElementById('videoGrid');
        
        // Modal elements
        this.videoModal = document.getElementById('videoModal');
        this.modalTitle = document.getElementById('modalTitle');
        this.modalVideo = document.getElementById('modalVideo');
        this.modalMetadata = document.getElementById('modalMetadata');
        this.modalClose = document.getElementById('modalClose');
        
        // Utility elements
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.toast = document.getElementById('toast');
    }
    
    bindEvents() {
        // Upload events
        this.uploadArea.addEventListener('click', () => this.videoInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        this.videoInput.addEventListener('change', this.handleFileSelect.bind(this));
        this.changeVideoBtn.addEventListener('click', this.resetUpload.bind(this));
        
        // Search events
        this.searchQuery.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !this.searchBtn.disabled) {
                this.handleSearch();
            }
        });
        this.searchBtn.addEventListener('click', this.handleSearch.bind(this));
        
        // Processing events
        this.clearLogsBtn.addEventListener('click', this.clearLogs.bind(this));
        
        // Modal events
        this.modalClose.addEventListener('click', this.closeModal.bind(this));
        this.videoModal.addEventListener('click', (e) => {
            if (e.target === this.videoModal) {
                this.closeModal();
            }
        });
        
        // Keyboard events
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }
    
    initializeSocket() {
        this.socket = io();
        
        this.socket.on('connected', (data) => {
            console.log('Connected to server:', data.message);
        });
        
        this.socket.on('log_update', (logEntry) => {
            this.addLogEntry(logEntry);
        });
        
        this.socket.on('processing_start', (data) => {
            this.processingText.textContent = `Processing "${data.query}" in ${data.video}...`;
        });
        
        this.socket.on('processing_complete', (data) => {
            this.processingText.textContent = `Found ${data.clips_found} clips for "${data.query}"`;
            this.showToast(`Processing complete! Found ${data.clips_found} clips.`, 'success');
        });
        
        this.socket.on('processing_error', (data) => {
            this.processingText.textContent = 'Processing failed';
            this.showToast(data.error, 'error');
        });
    }
    
    // File upload handlers
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }
    
    async handleFile(file) {
        // Validate file type
        const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 'video/x-msvideo', 'video/webm'];
        if (!allowedTypes.includes(file.type)) {
            this.showToast('Please select a valid video file (MP4, AVI, MOV, MKV, WEBM)', 'error');
            return;
        }
        
        // Validate file size (500MB)
        const maxSize = 500 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showToast('File size must be less than 500MB', 'error');
            return;
        }
        
        this.showLoadingOverlay(true);
        
        try {
            const formData = new FormData();
            formData.append('video', file);
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentVideo = result;
                this.displayVideoPreview(result);
                this.enableSearch();
                this.showToast('Video uploaded successfully!', 'success');
            } else {
                this.showToast(result.error || 'Upload failed', 'error');
            }
        } catch (error) {
            this.showToast('Network error during upload', 'error');
            console.error('Upload error:', error);
        } finally {
            this.showLoadingOverlay(false);
        }
    }
    
    displayVideoPreview(videoData) {
        this.uploadArea.style.display = 'none';
        this.videoPreview.style.display = 'block';
        
        this.videoName.textContent = videoData.filename;
        this.videoDuration.textContent = videoData.metadata.duration;
        this.videoResolution.textContent = videoData.metadata.resolution;
        this.videoFPS.textContent = videoData.metadata.fps;
        this.videoFrames.textContent = videoData.metadata.frames.toLocaleString();
    }
    
    resetUpload() {
        this.currentVideo = null;
        this.uploadArea.style.display = 'block';
        this.videoPreview.style.display = 'none';
        this.videoInput.value = '';
        this.disableSearch();
        this.hideResults();
        this.hideProcessing();
    }
    
    enableSearch() {
        this.searchQuery.disabled = false;
        this.topNSelect.disabled = false;
        this.searchBtn.disabled = false;
    }
    
    disableSearch() {
        this.searchQuery.disabled = true;
        this.topNSelect.disabled = true;
        this.searchBtn.disabled = true;
    }
    
    // Search functionality
    async handleSearch() {
        const query = this.searchQuery.value.trim();
        
        if (!query) {
            this.showToast('Please enter a search query', 'warning');
            return;
        }
        
        if (!this.currentVideo) {
            this.showToast('Please upload a video first', 'warning');
            return;
        }
        
        if (this.isProcessing) {
            this.showToast('Processing already in progress', 'warning');
            return;
        }
        
        this.isProcessing = true;
        this.showProcessing();
        this.hideResults();
        this.disableSearch();
        
        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    video_path: this.currentVideo.filepath,
                    query: query,
                    top_n: parseInt(this.topNSelect.value)
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result, query);
            } else {
                this.showToast(result.error || 'Search failed', 'error');
            }
        } catch (error) {
            this.showToast('Network error during search', 'error');
            console.error('Search error:', error);
        } finally {
            this.isProcessing = false;
            this.enableSearch();
        }
    }
    
    // Processing UI
    showProcessing() {
        this.processingSection.style.display = 'block';
        this.clearLogs();
    }
    
    hideProcessing() {
        this.processingSection.style.display = 'none';
    }
    
    addLogEntry(logEntry) {
        const logDiv = document.createElement('div');
        logDiv.className = 'log-entry';
        
        logDiv.innerHTML = `
            <span class="log-timestamp">${logEntry.timestamp}</span>
            <span class="log-level ${logEntry.level}">${logEntry.level}</span>
            <span class="log-message">${this.escapeHtml(logEntry.message)}</span>
        `;
        
        this.logsContent.appendChild(logDiv);
        this.logsContent.scrollTop = this.logsContent.scrollHeight;
    }
    
    clearLogs() {
        this.logsContent.innerHTML = '';
    }
    
    // Results display
    displayResults(result, query) {
        this.resultsSummary.innerHTML = `
            <h3>Found ${result.clips_found} relevant clips</h3>
            <p>Search query: "${query}"</p>
        `;
        
        this.videoGrid.innerHTML = '';
        
        result.output_videos.forEach((video, index) => {
            const videoCard = this.createVideoCard(video, index);
            this.videoGrid.appendChild(videoCard);
        });
        
        this.showResults();
    }
    
    createVideoCard(video, index) {
        const card = document.createElement('div');
        card.className = 'video-card';
        card.onclick = () => this.openVideoModal(video);
        
        card.innerHTML = `
            <div class="video-thumbnail">
                <video preload="metadata" muted>
                    <source src="/api/video/${video.filename}" type="video/mp4">
                </video>
                <div class="play-overlay">
                    <i class="fas fa-play"></i>
                </div>
            </div>
            <div class="video-details">
                <h3 class="video-title">Clip ${index + 1}</h3>
                <div class="video-metadata-grid">
                    <div class="video-metadata-item">
                        <span class="video-metadata-label">Duration</span>
                        <span class="video-metadata-value">${video.metadata.duration}</span>
                    </div>
                    <div class="video-metadata-item">
                        <span class="video-metadata-label">Resolution</span>
                        <span class="video-metadata-value">${video.metadata.resolution}</span>
                    </div>
                    <div class="video-metadata-item">
                        <span class="video-metadata-label">FPS</span>
                        <span class="video-metadata-value">${video.metadata.fps}</span>
                    </div>
                    <div class="video-metadata-item">
                        <span class="video-metadata-label">Score</span>
                        <span class="video-metadata-value">${video.metadata.score}</span>
                    </div>
                </div>
                ${video.metadata.score !== 'N/A' ? `<div class="score-badge">Relevance: ${video.metadata.score}</div>` : ''}
            </div>
        `;
        
        return card;
    }
    
    showResults() {
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    hideResults() {
        this.resultsSection.style.display = 'none';
    }
    
    // Modal functionality
    openVideoModal(video) {
        this.modalTitle.textContent = video.filename;
        this.modalVideo.src = `/api/video/${video.filename}`;
        
        this.modalMetadata.innerHTML = `
            <div class="metadata-item">
                <span class="metadata-label">Duration</span>
                <span class="metadata-value">${video.metadata.duration}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Resolution</span>
                <span class="metadata-value">${video.metadata.resolution}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">FPS</span>
                <span class="metadata-value">${video.metadata.fps}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Frames</span>
                <span class="metadata-value">${video.metadata.frames}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Relevance Score</span>
                <span class="metadata-value">${video.metadata.score}</span>
            </div>
        `;
        
        this.videoModal.classList.add('show');
        document.body.style.overflow = 'hidden';
    }
    
    closeModal() {
        this.videoModal.classList.remove('show');
        this.modalVideo.pause();
        this.modalVideo.src = '';
        document.body.style.overflow = '';
    }
    
    // Utility functions
    showLoadingOverlay(show) {
        if (show) {
            this.loadingOverlay.style.display = 'flex';
        } else {
            this.loadingOverlay.style.display = 'none';
        }
    }
    
    showToast(message, type = 'info') {
        const toast = this.toast;
        const toastIcon = toast.querySelector('.toast-icon');
        const toastMessage = toast.querySelector('.toast-message');
        
        // Set icon based on type
        let iconClass = 'fas fa-info-circle';
        if (type === 'success') iconClass = 'fas fa-check-circle';
        else if (type === 'error') iconClass = 'fas fa-exclamation-circle';
        else if (type === 'warning') iconClass = 'fas fa-exclamation-triangle';
        
        toastIcon.className = `toast-icon ${iconClass}`;
        toastMessage.textContent = message;
        
        // Reset classes and add type
        toast.className = `toast ${type}`;
        
        // Show toast
        toast.classList.add('show');
        
        // Hide after 5 seconds
        setTimeout(() => {
            toast.classList.remove('show');
        }, 5000);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VideoRAGApp();
});
