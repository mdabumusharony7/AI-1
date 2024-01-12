import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.applications.MobileNetV2(
    weights="imagenet", input_shape=(224, 224, 3), include_top=True
)


def generate_adversarial_example(image, epsilon=0.1):
    image_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        target_class = tf.keras.backend.argmax(prediction[0])
        loss = tf.keras.losses.sparse_categorical_crossentropy(target_class, prediction)

    gradient = tape.gradient(loss, image_tensor)
    signed_grad = tf.sign(gradient)
    perturbed_image = image + epsilon * signed_grad.numpy()
    perturbed_image = np.clip(perturbed_image, 0, 1)
    return perturbed_image



image_path = "path/to/your/image.jpg"
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0

epsilon = 0.1  
perturbed_img = generate_adversarial_example(img_array, epsilon=epsilon)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_array)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Adversarial Example (FGSM)")
plt.imshow(perturbed_img.squeeze())
plt.axis("off")

plt.tight_layout()
plt.show()
