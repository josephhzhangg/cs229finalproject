import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import util


# Define the Generator and Discriminator models
def make_generator_model(input_size, output_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=(input_size,), activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(output_size, activation='linear'))
    print("Generator Model created", model)
    return model


def make_discriminator_model(input_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, input_shape=(input_size,), activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    print("Discriminator Model created", model)
    return model


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    ### LOAD DATASET ###
    train_path = "train_split.csv"
    test_path = "test_split.csv"
    train_x, train_y = util.load_dataset(train_path, add_intercept=False)
    test_x, test_y = util.load_dataset(test_path, add_intercept=False)
    n_features = train_x.shape[1]

    # Define hyperparameters
    batch_size = 64
    num_epochs = 100
    z_dim = 10
    lr = 0.0002

    # Create generator and discriminator models
    generator = make_generator_model(z_dim, n_features)
    discriminator = make_discriminator_model(n_features)

    # Define loss function and optimizers
    criterion = tf.keras.losses.BinaryCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam(lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr)

    # Training loop
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        train_x, train_y = shuffle(train_x, train_y)  # Shuffle the training data
        num_batches = len(train_x) // batch_size
        for i in range(num_batches):
            # Train Discriminator with real data
            real_data = train_x[i * batch_size: (i + 1) * batch_size]
            real_labels = tf.ones((batch_size, 1))
            with tf.GradientTape() as tape:
                real_output = discriminator(real_data)
                real_loss = criterion(real_labels, real_output)
            grads = tape.gradient(real_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            # Train Discriminator with fake data
            z = tf.random.normal((batch_size, z_dim))
            with tf.GradientTape() as tape:
                fake_data = generator(z)
                fake_labels = tf.zeros((batch_size, 1))
                fake_output = discriminator(fake_data)
                fake_loss = criterion(fake_labels, fake_output)
            grads = tape.gradient(fake_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            discriminator_loss = real_loss + fake_loss

            # Train Generator
            z = tf.random.normal((batch_size, z_dim))
            with tf.GradientTape() as tape:
                fake_data = generator(z)
                fake_output = discriminator(fake_data)
                generator_loss = criterion(real_labels, fake_output)
            grads = tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        # Print training loss
        print(f"Epoch [{epoch + 1}/{num_epochs}] Discriminator Loss: {discriminator_loss:.4f}, Generator Loss: {generator_loss:.4f}")

    # Generate samples using the trained generator
    num_samples = 100
    z = tf.random.normal((num_samples, z_dim))
    generated_samples = generator(z).numpy()

    # Concatenate generated samples with original data
    combined_x = np.concatenate((train_x, generated_samples), axis=0)
    combined_y = np.concatenate((train_y, np.ones((num_samples,))), axis=0)

    # Train a classifier on the combined dataset
    classifier = LogisticRegression()
    classifier.fit(combined_x, combined_y)

    train_predictions = classifier.predict(combined_x)
    train_accuracy = accuracy_score(combined_y, train_predictions)
    train_f1 = f1_score(combined_y, train_predictions)
    train_recall = recall_score(combined_y, train_predictions)
    
    # Print evaluation metrics for training data
    print("Training Accuracy:", train_accuracy)
    print("Training F1 Score:", train_f1)
    print("Training Recall:", train_recall)

    # Evaluate the classifier on the test set
    test_predictions = classifier.predict(test_x)
    accuracy = accuracy_score(test_y, test_predictions)
    f1 = f1_score(test_y, test_predictions)
    recall = recall_score(test_y, test_predictions)
    print("Classification accuracy using GAN-generated samples: ", accuracy)
    print("F1 Score: ", f1)
    print("Recall: ", recall)


"""
# Save the generated samples to a file
np.savetxt(save_path, generated_samples, delimiter=',')
"""


if __name__ == '__main__':
    main()
