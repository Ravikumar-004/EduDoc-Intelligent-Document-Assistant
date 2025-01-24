import cv2
import numpy as np

# write code to generate video of character speaking the audio file.
import moviepy as mp

def generate_video_with_audio(character_image_path, audio_file_path, output_video_path):
    # Load the character image
    character_image = cv2.imread(character_image_path)
    height, width, layers = character_image.shape
    size = (width, height)

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('temp_video.mp4', fourcc, 30, size)

    # Load the audio file
    audio = mp.AudioFileClip(audio_file_path)
    duration = audio.duration

    # Generate video frames
    num_frames = int(duration * 30)  # Assuming 30 fps
    for _ in range(num_frames):
        video_writer.write(character_image)

    video_writer.release()

    # Combine video and audio
    video = mp.VideoFileClip('temp_video.mp4')
    video = video.with_audio(audio)
    video.write_videofile(output_video_path, codec='libx264')

# Example usage
generate_video_with_audio('character.png', 'audio.wav', 'output_video.mp4')
