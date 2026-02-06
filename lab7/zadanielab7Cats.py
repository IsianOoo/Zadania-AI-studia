import os
import zipfile
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras as k

ZIP_PATH = "cats.zip"
DATA_DIR = "cats_data"
IMG_SIZE = 64
BATCH = 64
LATENT = 128
EPOCHS = 30
SAVE_EVERY = 1
OUT_DIR = "gan_out"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if any(os.scandir(DATA_DIR)):
        return
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Brak {ZIP_PATH} w folderze")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)


def find_image_root():
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    best = None
    best_count = 0
    for root, _, files in os.walk(DATA_DIR):
        c = sum(1 for f in files if f.lower().endswith(exts))
        if c > best_count:
            best_count = c
            best = root
    if best is None or best_count == 0:
        raise RuntimeError("Nie znaleziono obrazów w rozpakowanych danych")
    return best


def make_dataset(path):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels=None,
        label_mode=None,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        shuffle=True,
    )
    ds = ds.map(lambda x: (tf.cast(x, tf.float32) / 127.5) - 1.0, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds


def build_generator():
    model = k.Sequential([
        k.layers.Input(shape=(LATENT,)),
        k.layers.Dense(4 * 4 * 512, use_bias=False),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(0.2),
        k.layers.Reshape((4, 4, 512)),

        k.layers.Conv2DTranspose(256, 4, strides=2, padding="same", use_bias=False),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(0.2),

        k.layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(0.2),

        k.layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False),
        k.layers.BatchNormalization(),
        k.layers.LeakyReLU(0.2),

        k.layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh"),
    ])
    return model


def build_discriminator():
    model = k.Sequential([
        k.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        k.layers.Conv2D(64, 4, strides=2, padding="same"),
        k.layers.LeakyReLU(0.2),
        k.layers.Dropout(0.3),

        k.layers.Conv2D(128, 4, strides=2, padding="same"),
        k.layers.LeakyReLU(0.2),
        k.layers.Dropout(0.3),

        k.layers.Conv2D(256, 4, strides=2, padding="same"),
        k.layers.LeakyReLU(0.2),
        k.layers.Dropout(0.3),

        k.layers.Conv2D(512, 4, strides=2, padding="same"),
        k.layers.LeakyReLU(0.2),
        k.layers.Dropout(0.3),

        k.layers.Flatten(),
        k.layers.Dense(1),
    ])
    return model


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def d_loss(real_logits, fake_logits):
    real = bce(tf.ones_like(real_logits), real_logits)
    fake = bce(tf.zeros_like(fake_logits), fake_logits)
    return real + fake


def g_loss(fake_logits):
    return bce(tf.ones_like(fake_logits), fake_logits)


def save_grid(gen, seed, epoch):
    os.makedirs(OUT_DIR, exist_ok=True)
    imgs = gen(seed, training=False)
    imgs = (imgs + 1.0) / 2.0
    imgs = tf.clip_by_value(imgs, 0.0, 1.0)

    n = int(np.sqrt(imgs.shape[0]))
    imgs = imgs[: n * n]

    rows = []
    for r in range(n):
        row = tf.concat([imgs[r * n + c] for c in range(n)], axis=1)
        rows.append(row)
    grid = tf.concat(rows, axis=0)

    path = os.path.join(OUT_DIR, f"epoch_{epoch:03d}.png")
    k.utils.save_img(path, grid)


@tf.function
def train_step(images, gen, disc, g_opt, d_opt):
    noise = tf.random.normal([tf.shape(images)[0], LATENT])

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake = gen(noise, training=True)

        real_logits = disc(images, training=True)
        fake_logits = disc(fake, training=True)

        dl = d_loss(real_logits, fake_logits)
        gl = g_loss(fake_logits)

    d_grads = d_tape.gradient(dl, disc.trainable_variables)
    g_grads = g_tape.gradient(gl, gen.trainable_variables)

    d_opt.apply_gradients(zip(d_grads, disc.trainable_variables))
    g_opt.apply_gradients(zip(g_grads, gen.trainable_variables))

    return dl, gl


def main():
    ensure_data()
    root = find_image_root()
    ds = make_dataset(root)

    gen = build_generator()
    disc = build_discriminator()

    g_opt = k.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    d_opt = k.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    seed = tf.random.normal([16, LATENT])

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        d_losses = []
        g_losses = []

        for batch in ds:
            dl, gl = train_step(batch, gen, disc, g_opt, d_opt)
            d_losses.append(float(dl))
            g_losses.append(float(gl))

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"D: {np.mean(d_losses):.4f} | G: {np.mean(g_losses):.4f} | "
            f"time: {time.time()-t0:.1f}s"
        )

        if epoch % SAVE_EVERY == 0:
            save_grid(gen, seed, epoch)

    gen.save(os.path.join(OUT_DIR, "generator.keras"))
    disc.save(os.path.join(OUT_DIR, "discriminator.keras"))


if __name__ == "__main__":
    main()
