import os
from PIL import Image

def convert_jpg_to_png(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)

    # Iterate through each file
    for file_name in files:
        if file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
            # Open the image file
            image_path = os.path.join(input_folder, file_name)
            with Image.open(image_path) as img:
                # Convert JPG to PNG format
                png_file_name = os.path.splitext(file_name)[0] + '.png'
                output_path = os.path.join(output_folder, png_file_name)
                img.save(output_path, 'PNG')
                print(f"{file_name} converted to {png_file_name}")

# Specify the input and output folders
input_folder = './'
output_folder = './output'

# Convert JPG to PNG
convert_jpg_to_png(input_folder, output_folder)
