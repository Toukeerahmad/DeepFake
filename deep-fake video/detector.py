import cv2
from deepface import DeepFace

def detect(video_path, frame_skip=5, enforce_detection=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0, 0

    deepfake_frames = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

     
        if total_frames % frame_skip != 0:
            total_frames += 1
            continue

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            analysis = DeepFace.analyze(
            rgb_frame,
            actions=["emotion"],
            enforce_detection=False  
            )
            dominant_emotion = analysis[0]["dominant_emotion"]
            emotion_score = analysis[0]["emotion"].get(dominant_emotion, 0)

            # giving heristic values to frame
            if dominant_emotion == "neutral" and emotion_score > 90:  
                deepfake_frames += 1

        except Exception as e:
            print(f"Skipping frame due to error: {e}")

        total_frames += 1

    cap.release()
    return deepfake_frames, total_frames



video_file = "virat.mp4"  
deepfake_frames, total_frames = detect(video_file)

if total_frames == 0:
    print("No frames were processed. The video might be empty or corrupted.")
else:
    
    if deepfake_frames >= 1: 
        print("Found with some of the fake frames so, the video is likely to be a deepfake video.!!")
    else:
        print("since no fake frames found so, the video is likely to be authentic.")
