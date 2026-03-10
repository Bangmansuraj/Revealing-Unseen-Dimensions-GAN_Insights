import tensorflow as tf
import os
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg') # Use an interactive backend
from matplotlib import pyplot as plt

print("TensorFlow Version:", tf.__version__)

# --- Define the paths to our image folders ---
PATH = "C:/Users/suraj ns/Downloads/major_project/"
FRONT_VIEW_PATH = os.path.join(PATH, 'view_front')
SIDE_VIEW_PATH = os.path.join(PATH, 'view_side')

# --- Define image dimensions ---
IMG_WIDTH = 256
IMG_HEIGHT = 256

# --- Data Loading and Processing Functions ---
def load_image(image_file):
    front_view = tf.io.read_file(image_file)
    front_view = tf.io.decode_jpeg(front_view)
    side_view_path = tf.strings.regex_replace(image_file, 'view_front', 'view_side')
    side_view = tf.io.read_file(side_view_path)
    side_view = tf.io.decode_jpeg(side_view)
    front_view = tf.cast(front_view, tf.float32)
    side_view = tf.cast(side_view, tf.float32)
    return front_view, side_view

def resize(front_view, side_view, height, width):
    front_view = tf.image.resize(front_view, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    side_view = tf.image.resize(side_view, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return front_view, side_view

def normalize(front_view, side_view):
    front_view = (front_view / 127.5) - 1
    side_view = (side_view / 127.5) - 1
    return front_view, side_view

def load_image_train(image_file):
    front_view, side_view = load_image(image_file)
    front_view, side_view = resize(front_view, side_view, IMG_HEIGHT, IMG_WIDTH)
    front_view, side_view = normalize(front_view, side_view)
    return front_view, side_view

# --- Generator (U-Net) Model ---
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

# --- Discriminator Model ---
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

# --- Instantiate Models ---
generator = Generator()
discriminator = Discriminator()

# --- Loss Functions and Optimizers ---
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 100

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + (LAMBDA * l1_loss)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# --- Checkpointing ---
checkpoint_dir = os.path.join(PATH, 'training_checkpoints') # Use full path
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

# --- Image Generation for Testing ---
def generate_images(model, test_input, target, epoch):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 5))
    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    
    # Use the main PATH variable to create the folder in the right place
    output_folder = os.path.join(PATH, 'training_images')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plt.savefig(os.path.join(output_folder, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()

# --- Training Step Function ---
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

# --- Main Training Loop ---
def fit(train_ds, test_ds, steps, epochs):
    example_input, example_target = next(iter(test_ds.take(1)))
    for epoch in range(epochs):
        start = time.time()
        for n, (input_image, target) in train_ds.take(steps).enumerate():
            print(f"Epoch {epoch+1}, Step {n+1}/{steps}", end='\r')
            train_step(input_image, target)
        print()
        generate_images(generator, example_input, example_target, epoch + 1)
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print(f'Time taken for epoch {epoch + 1} is {time.time()-start:.2f} sec')

# --- Prepare Dataset and Start Training ---
BUFFER_SIZE = 400
BATCH_SIZE = 1
EPOCHS = 100 # You can set this higher for the full run

train_dataset = tf.data.Dataset.list_files(FRONT_VIEW_PATH + '/*.png')
train_dataset = train_dataset.take(2000) # Use only 2000 images
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

test_dataset = train_dataset
steps_per_epoch = 2000

fit(train_dataset, test_dataset, steps_per_epoch, epochs=EPOCHS)