import cv2
import numpy as np
import sys
import tensorflow as tf

IMG_WIDTH = 30
IMG_HEIGHT = 30

CATEGORY_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}


def main():

    if len(sys.argv) != 3:
        sys.exit("Usage: python predict.py model.keras image_path")

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    model = tf.keras.models.load_model(model_path)

    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"Could not read image: {image_path}")

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = np.array([image])

    predictions = model.predict(image)
    predicted_category = int(np.argmax(predictions))
    confidence = np.max(predictions)
    category_name = CATEGORY_NAMES.get(predicted_category, "Unknown")

    print(f"Predicted category: {predicted_category} ({category_name})")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()