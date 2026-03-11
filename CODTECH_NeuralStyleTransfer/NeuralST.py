import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import os



def load_and_prep_image(path):
    print(f"-> Loading: {path}")
    max_dim = 512
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    return img[tf.newaxis, :]

def save_result(tensor, filename):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    result_img = PIL.Image.fromarray(tensor)
    result_img.save(filename)
    print(f"-> Success! Saved as {filename}")


def run_style_transfer(my_photo_path, art_style_path):
    try:
        content_img = load_and_prep_image(my_photo_path) #
        style_img = load_and_prep_image(art_style_path)   #

        print("-> Downloading/Loading the NST model... hang tight.")
        nst_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2') #

        
        print("-> Applying style... this usually takes 5-10 seconds on a standard CPU.")
        stylized_output = nst_model(tf.constant(content_img), tf.constant(style_img))[0] #

        save_result(stylized_output, 'stylized_result.jpg')

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np.squeeze(content_img))
        plt.title('Original Photo')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(stylized_output))
        plt.title('Stylized Result')
        plt.axis('off')
        plt.show() #

    except Exception as e:
        print(f"Error: {e}. Check your file paths or internet connection.")

if __name__ == "__main__":
    run_style_transfer('my_content_photo.jpg', 'starry_night_style.jpg')