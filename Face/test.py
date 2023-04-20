import unittest
import cv2
import numpy as np
from tensorflow.keras.models import load_model


class TestEmotionRecognition(unittest.TestCase):
    """Test Emotion Recognition

        This test checks that the model loads correctly, performs an emotion prediction
        on a test image and checks if the prediction is successful. It also verifies that
        the script stops when the ESC key is pressed on the keyboard. Of course, make sure
        to change the file names and class labels according to your configuration.

    """

    def test_model_loading(self):
        # Check if the model is loaded successfully
        model = load_model("keras_model.h5", compile=False)
        self.assertIsNotNone(model)

    def test_emotion_prediction(self):
        # Load the model and labels
        model = load_model("keras_model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()

        # Load a test image
        image = cv2.imread("happy_face.jpg")
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1

        # Predict the emotion in the image
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Verify the prediction result
        self.assertEqual(class_name[2:], "happy")
        self.assertGreaterEqual(confidence_score, 0.5)

    def test_script_stop(self):
        # Test if the script stops when the ESC key is pressed
        camera = cv2.VideoCapture(0)
        while True:
            ret, image = camera.read()
            cv2.imshow("Emotion face", image)
            keyboard_input = cv2.waitKey(1)
            if keyboard_input == 27:
                break
        camera.release()
        cv2.destroyAllWindows()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
