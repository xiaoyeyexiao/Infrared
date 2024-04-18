import cv2
import os

def video_to_image(video_path, output_dir, frame_interval=10, isoneimage = False):
    """
    Convert video to images with specified frame interval.
    
    Args:
    - video_path: Path to the input video file.
    - output_dir: Directory to save the output images.
    - frame_interval: Interval between frames to be saved as images.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    frame_count = 0
    image_index = 1

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        frame_count += 1
        
        # Save frame as image every frame_interval frames
        if frame_count % frame_interval == 0:
            # 在字符串前加上 f 或 F 前缀，可以在字符串中直接使用大括号 {} 来引用变量或表达式的值
            image_path = os.path.join(output_dir, f"frame_{image_index}.jpg")
            cv2.imwrite(image_path, frame)
            # print(f"Saved frame {image_index}")
            image_index += 1
            
            if isoneimage:
                exit()

    # Release the video capture object
    cap.release()
