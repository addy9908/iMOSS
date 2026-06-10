import base64
import io
from PIL import Image

def generate_pre_resized_logo_module():
    """
    Resizes the logo first, then encodes the smaller image to Base64,
    and saves it to a Python module.
    """
    image_path = 'iMOSS-V_new.png'
    output_file = 'logo_module.py'
    final_size = (200, 40)

    try:
        # Open the original large image
        with Image.open(image_path) as img:
            # Resize it
            resized_img = img.resize(final_size, Image.LANCZOS)

            # Save the resized image to an in-memory stream instead of a file
            with io.BytesIO() as output_stream:
                resized_img.save(output_stream, format="PNG")
                # Get the binary data of the *resized* image
                resized_image_bytes = output_stream.getvalue()

        # Encode the (now much smaller) binary data to a Base64 string
        encoded_string = base64.b64encode(resized_image_bytes).decode('utf-8')

        # Write the module file with the smaller string
        with open(output_file, "w") as f:
            f.write("import base64\n")
            f.write("import io\n")
            f.write("from PIL import Image\n\n")
            f.write(f'# Base64 for a {final_size[0]}x{final_size[1]} image\n')
            f.write(f'LOGO_BASE64 = "{encoded_string}"\n\n')
            f.write("def get_logo():\n")
            f.write("    image_data = base64.b64decode(LOGO_BASE64)\n")
            f.write("    image_stream = io.BytesIO(image_data)\n")
            f.write("    img = Image.open(image_stream)\n")
            f.write("    img.load()\n")
            f.write("    return img\n")

        print(f"SUCCESS! {output_file} has been created with a pre-resized logo.")

    except FileNotFoundError:
        print(f"Error: Could not find {image_path}. Make sure it is in the same folder.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_pre_resized_logo_module()
