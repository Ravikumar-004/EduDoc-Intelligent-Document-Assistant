from PIL import Image, ImageDraw

def create_character_images():
    # Image size
    width, height = 400, 400

    # Common settings
    face_color = (255, 224, 189)   # Light skin tone color
    eye_color = (0, 0, 0)          # Black color for eyes
    mouth_color = (150, 0, 0)      # Dark red color for mouth
    background_color = (255, 255, 255)  # White background

    # Create the closed mouth image
    closed_img = Image.new('RGB', (width, height), color=background_color)
    draw_closed = ImageDraw.Draw(closed_img)

    # Draw face (circle)
    draw_closed.ellipse([(50, 50), (350, 350)], fill=face_color, outline=(0, 0, 0), width=2)

    # Draw eyes
    # Left eye
    draw_closed.ellipse([(130, 150), (170, 190)], fill=eye_color)
    # Right eye
    draw_closed.ellipse([(230, 150), (270, 190)], fill=eye_color)

    # Draw closed mouth (line)
    draw_closed.line([(150, 260), (250, 260)], fill=mouth_color, width=5)

    # Save the closed mouth image
    closed_img.save('character_closed.png')

    # Create the open mouth image
    open_img = Image.new('RGB', (width, height), color=background_color)
    draw_open = ImageDraw.Draw(open_img)

    # Draw face (circle)
    draw_open.ellipse([(50, 50), (350, 350)], fill=face_color, outline=(0, 0, 0), width=2)

    # Draw eyes
    # Left eye
    draw_open.ellipse([(130, 150), (170, 190)], fill=eye_color)
    # Right eye
    draw_open.ellipse([(230, 150), (270, 190)], fill=eye_color)

    # Draw open mouth (ellipse)
    draw_open.ellipse([(170, 240), (230, 280)], fill=mouth_color)

    # Save the open mouth image
    open_img.save('character_open.png')

# Generate the images
create_character_images()
