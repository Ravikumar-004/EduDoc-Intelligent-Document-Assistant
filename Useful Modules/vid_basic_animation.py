from moviepy import *
import numpy as np

def generate_speaking_video(closed_mouth_img_path, open_mouth_img_path, audio_path, output_path):
    # Load audio
    audio = AudioFileClip(audio_path)
    
    # Get the duration of the audio
    duration = audio.duration
    
    # Set frames per second
    fps = 24
    
    # Load images
    img_closed = ImageClip(closed_mouth_img_path).with_duration(0.30)
    img_open = ImageClip(open_mouth_img_path).with_duration(0.15)
    img_closed_op = ImageClip(closed_mouth_img_path).with_duration(0.15)
    
    # Generate clips based on volume
    clips = []
    for t in np.arange(0, duration, 0.30):
        volume = audio.subclipped(t, t + 0.30).to_soundarray().mean()
        if volume > 0:
            clips.append(img_open)
            clips.append(img_closed_op)
        else:
            clips.append(img_closed)
    
    # Concatenate all clips
    video = concatenate_videoclips(clips, method='compose')
    
    # Set audio to the video
    video = video.with_audio(audio)
    
    # Write the resulting video file
    video.write_videofile(output_path, fps=fps)

# Parameters
closed_mouth_img_path = 'EduAssets/character_closed.png'  # Path to closed mouth image
open_mouth_img_path = 'EduAssets/character_open.png'      # Path to open mouth image
audio_path = 'test_aud.wav'                               # Path to your audio file
output_path = 'output_video_basic_animation.mp4'          # Output video path

# Generate the video
generate_speaking_video(closed_mouth_img_path, open_mouth_img_path, audio_path, output_path)
