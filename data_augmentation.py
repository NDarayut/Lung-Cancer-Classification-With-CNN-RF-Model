from PIL import Image
import os

def flip_image_horizontally(image_path, save_path):
  """
  Flips an image horizontally and saves it to a specified path.

  Args:
      image_path: Path to the original image file.
      save_path: Path to save the flipped image.
  """
  try:
    # Open the image
    img = Image.open(image_path)

    # Get image dimensions
    width, height = img.size

    # Create a new image to hold the flipped data
    flipped_image = Image.new(img.mode, (width, height))

    # Loop through each pixel and flip horizontally
    for y in range(height):
      for x in range(width):
        # Get pixel from original image
        pixel = img.getpixel((x, y))

        # Set the corresponding pixel in the flipped image
        flipped_image.putpixel((width - x - 1, y), pixel)

    # Save the flipped image
    flipped_image.save(save_path)
    print(f"Image flipped and saved: {save_path}")
  except Exception as e:
    print(f"Error flipping image: {e}")

for i in range(0, len(os.listdir('D:\\Cancer Detection using ML\\The IQ-OTHNCCD lung cancer dataset\\Bengin cases'))):
  # Example usage (replace with your actual paths)
  image_path = f"D:\\Cancer Detection using ML\\The IQ-OTHNCCD lung cancer dataset\\Bengin cases\\Bengin case ({i+1}).jpg"
  save_path = f"D:\\Cancer Detection using ML\\Data augmented\\Bengin\\Bengin case ({i+121}).jpg"
  flip_image_horizontally(image_path, save_path)
