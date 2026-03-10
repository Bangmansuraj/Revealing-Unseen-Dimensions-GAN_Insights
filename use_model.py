import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg') # Use an interactive backend
from matplotlib import pyplot as plt

# --- IMPORTANT: We must include the model architecture definition ---
# --- so TensorFlow knows how to load the saved weights. ---
# --- Copy/paste these from your training script. ---

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [downsample(64, 4, False), downsample(128, 4), downsample(256, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4)]
    up_stack = [upsample(512, 4, True), upsample(512, 4, True), upsample(512, 4, True), upsample(512, 4), upsample(256, 4), upsample(128, 4), upsample(64, 4)]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# --- Define the paths ---
PATH = "C:/Users/suraj ns/Downloads/major_project/"
CHECKPOINT_DIR = os.path.join(PATH, 'training_checkpoints')

# --- ✍️ ACTION REQUIRED: Set the path to the image you want to test ---
# --- You can pick any image from your 'view_front' folder ---
TEST_IMAGE_PATH = "C:/Users/suraj ns/Downloads/major_project/chair image.jpeg"
# --- Instantiate the Generator and its optimizer ---
generator = Generator()
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# --- Create a checkpoint object and restore the latest training ---
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator)
checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).expect_partial()
print("✅ Model restored from the latest checkpoint.")


# --- Function to load and prepare a single image for inference ---
# (This is the replacement for your load_and_process_image function)

def load_and_process_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    # --- THE FIX IS HERE ---
    img = tf.cast(img, tf.float32) # Convert from integers to floats
    
    img = (img / 127.5) - 1 # Now the math will work
    img = tf.expand_dims(img, axis=0) # Add a batch dimension
    return img

# --- Load the test image and generate a prediction ---
test_input_image = load_and_process_image(TEST_IMAGE_PATH)
prediction = generator(test_input_image, training=False)
print("✅ Prediction generated.")

# --- Display the results ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Your Input Image")
plt.imshow(test_input_image[0] * 0.5 + 0.5) # De-normalize for display
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("AI Generated Side View")
plt.imshow(prediction[0] * 0.5 + 0.5) # De-normalize for display
plt.axis('off')

plt.show()