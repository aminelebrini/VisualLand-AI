import tensorflow as tf

def predict_single_image(image_path):

    class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    model = tf.keras.models.load_model("models/intel_image_classifier_model.keras")

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)

    predictions = model.predict(image)
    # print(predictions)

    predicted_label = tf.argmax(predictions[0]).numpy()
    predicted_class_name = class_names[predicted_label]
    confidence_score = float(tf.reduce_max(predictions[0])) * 100
    print(f"Predicted class: {predicted_class_name}, Confidence: {confidence_score:.2f}%")


    return predicted_class_name, confidence_score
