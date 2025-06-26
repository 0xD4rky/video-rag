import gradio as gr
import asyncio
import os
from pathlib import Path
import sys

sys.path.append('/Users/darky/Documents/video-rag/src')

from src.agent import VideoSearchAgent


class GradioVideoRAG:
    def __init__(self):
        self.agent = VideoSearchAgent()
        self.output_dir = "/Users/darky/Documents/video-rag/data/output"
        self.setup_directories()
    
    def setup_directories(self):
        Path("/Users/darky/Documents/video-rag/data/faiss").mkdir(parents=True, exist_ok=True)
        Path("/Users/darky/Documents/video-rag/data/output").mkdir(parents=True, exist_ok=True)
        Path("/Users/darky/Documents/video-rag/data/logs").mkdir(parents=True, exist_ok=True)
    
    def clear_previous_outputs(self):
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    
    def search_video(self, video_file, query, top_n):
        if not video_file or not query:
            return "Please provide both a video file and a search query.", []
        
        self.clear_previous_outputs()
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            clips = loop.run_until_complete(self.agent.run(video_file, query, int(top_n)))
            loop.close()
            
            if clips:
                result_text = f"Found {len(clips)} relevant clips for query: '{query}'"
                return result_text, clips
            else:
                return "No relevant clips found for the given query.", []
                
        except Exception as e:
            return f"Error processing video: {str(e)}", []


def create_interface():
    video_rag = GradioVideoRAG()
    
    interface = gr.Interface(
        fn=video_rag.search_video,
        inputs=[
            gr.File(label="Upload Video File", file_types=[".mp4", ".avi", ".mov", ".mkv"]),
            gr.Textbox(label="Search Query", placeholder="Describe what you're looking for in the video..."),
            gr.Slider(label="Number of clips to extract", minimum=1, maximum=10, value=3, step=1)
        ],
        outputs=[
            gr.Textbox(label="Search Results"),
            gr.Gallery(label="Found Video Clips")
        ],
        title="Video RAG Search",
        description="Upload a video and retrieve the relevant clips you desire just by a querying",
        article="""
        ### How it works:
        1. Upload a video file (MP4, AVI, MOV, MKV)
        2. Enter a query describing what you're looking for
        3. Choose how many clips you want to extract
        4. Click Submit to find relevant scenes
        
        The system will analyze the video, extract scenes, and return the most relevant clips based on your query.
        """
    )
    
    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )