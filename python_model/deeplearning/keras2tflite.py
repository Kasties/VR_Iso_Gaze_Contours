import tensorflow as tf
import os
import argparse

from countour_encoder_prediction import loadModel

def main(args):
    print("Load model")
    contour_model = loadModel(args.modelpath)
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(contour_model)
    print("Convert model")
    tflite_model = converter.convert()

    # Save the model.
    print("Save tflite model")
    with open(os.path.join(args.modelpath, 'contour.tflite'), 'wb') as f:
      f.write(tflite_model)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, required=True) #folder where json and checkpoint are saved
    args = parser.parse_args()
    main(args)