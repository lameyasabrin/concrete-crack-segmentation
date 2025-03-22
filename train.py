import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, precision_recall_curve

# Import image loading and augmentation functions
from load_image import read_image, read_mask_image, get_train_image, get_mask_image

# Import the NestedModel class from model.py
from model import NestedModel


def plot_training_statistics(history):
    """
    Plot training statistics such as loss, accuracy, dice coefficient, and validation loss if available.
    """
    plt.figure(figsize=(20, 14))
    plt.suptitle('Training Statistics on Train Set')
    
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], 'r')
    plt.title('Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], 'g')
    plt.title('Accuracy')
    
    if 'dice_coe' in history.history:
        plt.subplot(2, 2, 3)
        plt.plot(history.history['dice_coe'], 'b')
        plt.yticks(np.arange(0.0, 1.0, 0.10))
        plt.title('Dice Coefficient')
    
    if 'val_loss' in history.history:
        plt.subplot(2, 2, 4)
        plt.plot(history.history['val_loss'], 'm')
        plt.title('Validation Loss')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    # --- File paths ---
    base_dir = "G:/data"
    train_path = os.path.join(base_dir, "trainx")
    train_mask_path = os.path.join(base_dir, "trainy")
    test_path = os.path.join(base_dir, "testx")
    test_mask_path = os.path.join(base_dir, "testy")
    auc_out_dir = base_dir  # For saving ROC/PR plots if needed
    img_out_dir = os.path.join(base_dir, "segmented")

    # --- Load Images ---
    print("Loading training and testing images...")
    train_img = read_image(train_path)
    train_mask_img = read_mask_image(train_mask_path)
    test_img = read_image(test_path)
    test_mask_img = read_mask_image(test_mask_path)
    
    # --- Data Augmentation ---
    train = get_train_image(train_img, augmentation=True)
    train_mask = get_mask_image(train_mask_img, augmentation=True)
    
    # Convert test images to numpy arrays
    test = np.array(test_img)
    test_mask = np.array(test_mask_img)
    
    # Expand dimensions for training masks if necessary (assuming single-channel)
    train_mask = np.expand_dims(train_mask, axis=3)
    
    # --- Prepare Evaluation Data ---
    # Flatten the test mask and normalize it for ROC and PR curves
    y_true = test_mask.ravel() / 255.0

    # --- Training Parameters ---
    batch_size = 16
    filter_size = 32
    init_lr = 2e-4
    N_epochs = 1  # Increase for full training
    img_height, img_width = 32, 32
    img_size = (img_height, img_width, 3)
    
    # --- Build and Compile the Model using NestedModel ---
    print("Building the model...")
    nested_model_instance = NestedModel(img_size, filter_size, init_lr)
    model = nested_model_instance.build_model()
    model.summary()
    
    # --- Set Up Model Checkpoint ---
    checkpoint_filepath = 'weights.h5'
    checkpointer = ModelCheckpoint(filepath=checkpoint_filepath,
                                   verbose=1,
                                   monitor='val_loss',
                                   mode='auto',
                                   save_best_only=True)
    
    # --- Train the Model ---
    print("Training the model...")
    history = model.fit(train,
                        train_mask,
                        batch_size=batch_size,
                        epochs=N_epochs,
                        validation_split=0.1,  # 10% of training data for validation
                        callbacks=[checkpointer])
    
    # Plot training statistics
    plot_training_statistics(history)
    
    # --- Predict on a Single Test Image ---
    img_num = 29  # Change index as needed
    if img_num < len(test_img):
        img_input = test_img[img_num].reshape(1, img_height, img_width, 3)
        img_pred = model.predict(img_input)
    
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(test_img[img_num].reshape(img_height, img_width, 3))
        plt.title('Original Image')
    
        plt.subplot(1, 2, 2)
        plt.imshow(img_pred.reshape(img_height, img_width), cmap='binary')
        plt.title('Predicted Output')
        plt.show()
    else:
        print("The specified img_num exceeds the number of test images.")
    
    # --- Predict on the Entire Test Set ---
    print("Predicting on the test set...")
    pred = model.predict(test, batch_size=64, verbose=0)
    pred = pred.ravel()
    
    # Threshold the test mask to generate binary labels
    pixel_threshold = 0.5
    mask_test = np.where(test_mask >= pixel_threshold, 1, 0)
    mm = mask_test.ravel()
    
    # Save true labels to CSV for later analysis if needed
    true_label_csv = os.path.join(base_dir, "true_label.csv")
    np.savetxt(true_label_csv, mm, delimiter=",")
    print(f"True labels saved to {true_label_csv}")
    
    # --- Plot ROC Curve ---
    fpr, tpr, thresholds = roc_curve(mm, pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b', label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    # --- Plot Precision-Recall Curve ---
    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(mm, pred)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, 'g', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    plt.show()


# if __name__ == '__main__':
#     main()



#yG+IRhm1uiIUANy+mt/RTtOBVKujsw0kP47vc4IX4Ug
