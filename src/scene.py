import cv2
from PIL import Image



def extract_scenes(video_path, scene_duration=2, fps=5):

    """
    This function is for extracting the scenes from the video.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / video_fps
    
    scenes = []
    for start_time in range(0, int(duration), scene_duration):
        end_time = min(start_time + scene_duration, duration)
        frames = []
        frame_indices = []
        
        for t in range(int(start_time * video_fps), int(end_time * video_fps), video_fps // fps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                frame_indices.append(t)
        
        if frames:
            scenes.append((start_time, end_time, frames, frame_indices))

    cap.release()
    return scenes, video_fps

def save_scene_video(video_path, start_time, end_time, output_path):

    """
    A function for saving the relevant retrieved scenes
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    while cap.get(cv2.CAP_PROP_POS_MSEC) < end_time * 1000:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()

