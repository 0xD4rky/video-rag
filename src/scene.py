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