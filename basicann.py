#!/usr/bin/env python3
"""
Simple MNIST Machine Learning Demo

This script:
    - Loads the MNIST handwritten digits dataset
    - Preprocesses the data (normalizes pixel values, reshapes)
    - Builds a small neural network using TensorFlow/Keras
    - Trains the model and evaluates it on test data
    - Shows example predictions

It is designed for students learning Python and basic machine learning.
"""

import sys
import traceback

# Try to import TensorFlow and Keras.
# If it's not installed, we show a friendly error message.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError as e:
    print("ERROR: TensorFlow is not installed.")
    print("Install it with: pip install tensorflow")
    sys.exit(1)


# Some constants (these could be changed by students)
EPOCHS = 3           # How many times we show all training data to the model
BATCH_SIZE = 128     # How many samples per training step
RANDOM_SEED = 42     # For reproducibility (makes results similar each run)


def load_mnist_data():
    """
    Load the MNIST dataset from Keras and split into train/test sets.

    Returns:
        (x_train, y_train), (x_test, y_test)
        where x_* are images and y_* are labels.
    """
    try:
        # Keras can download the dataset automatically if not present.
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        print("Loaded MNIST dataset successfully.")
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        print("ERROR: Failed to load MNIST dataset.")
        # Print the actual error for debugging/learning purposes
        print("Details:", e)
        sys.exit(1)


def preprocess_data(x_train, y_train, x_test, y_test):
    """
    Preprocess the raw MNIST data so it is ready for the neural network.

    Steps:
        - Convert pixel values from integers (0-255) to floats (0.0-1.0)
        - Flatten each 28x28 image into a 784-dimensional vector
        - Leave labels as integers (0-9)

    Returns:
        Preprocessed (x_train, y_train, x_test, y_test)
    """
    try:
        # Convert from uint8 [0, 255] to float32 [0.0, 1.0]
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # Remember the original shape (num_samples, 28, 28)
        # We want to reshape each image to (784,) = 28 * 28
        num_pixels = 28 * 28
        x_train = x_train.reshape((-1, num_pixels))
        x_test = x_test.reshape((-1, num_pixels))

        print("Preprocessing complete.")
        print("x_train shape:", x_train.shape)
        print("x_test shape:", x_test.shape)

        return x_train, y_train, x_test, y_test
    except Exception as e:
        print("ERROR: Failed to preprocess data.")
        print("Details:", e)
        sys.exit(1)


def build_model(input_shape, num_classes):
    """
    Build a simple feedforward neural network for MNIST.

    Architecture:
        - Input layer (size 784)
        - Dense hidden layer with 128 neurons and ReLU activation
        - Dense hidden layer with 64 neurons and ReLU activation
        - Output layer with 'num_classes' neurons and softmax activation

    Args:
        input_shape: integer, the size of the input vector (784 for MNIST).
        num_classes: integer, number of output classes (10 for digits 0-9).

    Returns:
        A compiled Keras model ready to be trained.
    """
    try:
        model = keras.Sequential(
            [
                layers.Input(shape=(input_shape,)),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Compile the model: tell it how to measure loss and which optimizer to use
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",  # labels are integers 0-9
            metrics=["accuracy"],
        )

        print("Model built and compiled successfully.")
        model.summary()  # Show a summary so students can see the layers
        return model
    except Exception as e:
        print("ERROR: Failed to build/compile the model.")
        print("Details:", e)
        sys.exit(1)


def train_model(model, x_train, y_train, x_test, y_test):
    """
    Train the model on training data and evaluate on test data.

    Args:
        model: compiled Keras model.
        x_train, y_train: training data and labels.
        x_test, y_test: test data and labels.

    Returns:
        The trained model.
    """
    try:
        # Set random seeds to try to make results repeatable
        tf.random.set_seed(RANDOM_SEED)

        print("Starting training...")
        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,  # Use 10% of training data as validation
            verbose=2,             # 1 or 2 gives some progress output
        )

        print("\nTraining complete. Evaluating on test data...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")

        return model
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
        # Still return the model, even if partially trained
        return model
    except Exception as e:
        print("ERROR: An error occurred during training or evaluation.")
        print("Details:", e)
        traceback.print_exc()
        sys.exit(1)


def show_example_predictions(model, x_test, y_test, num_examples=5):
    """
    Show example predictions from the trained model.

    Args:
        model: trained Keras model.
        x_test, y_test: test data and labels.
        num_examples: how many example predictions to print.
    """
    try:
        # Use the first few samples from the test set
        examples = x_test[:num_examples]
        true_labels = y_test[:num_examples]

        # Model outputs probability distribution over classes
        predictions = model.predict(examples, verbose=0)

        print(f"\nShowing {num_examples} example predictions:")
        for i in range(num_examples):
            probs = predictions[i]
            predicted_label = probs.argmax()  # index of highest probability
            confidence = probs[predicted_label]

            print(f"Example {i + 1}:")
            print(f"  True label      : {true_labels[i]}")
            print(f"  Predicted label : {predicted_label}")
            print(f"  Confidence      : {confidence:.2f}")
    except Exception as e:
        print("ERROR: Failed to generate example predictions.")
        print("Details:", e)


def main():
    """
    Main function that ties everything together.
    """
    print("MNIST Machine Learning Demo")

    # Load the data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Preprocess
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )

    # Build model
    input_shape = x_train.shape[1]   # 784
    num_classes = 10                 # digits 0-9
    model = build_model(input_shape, num_classes)

    # Train & evaluate
    model = train_model(model, x_train, y_train, x_test, y_test)

    # Show some example predictions
    show_example_predictions(model, x_test, y_test, num_examples=5)

    print("\nDone. You can now try modifying the model or parameters!")


if __name__ == "__main__":
    # Wrap main() in a try/except so we catch unexpected errors nicely.
    try:
        main()
    except Exception as e:
        print("A fatal error occurred in the program.")
        print("Details:", e)
        traceback.print_exc()
        sys.exit(1)
