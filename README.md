# Speaker Identification System Using GMM and CNN

This project implements a **speaker identification system** using two complementary approaches: **Gaussian Mixture Models (GMM)** and **Convolutional Neural Networks (CNNs)**. The system processes audio recordings to classify speakers based on their unique vocal characteristics. Both methods utilize **Mel Frequency Cepstral Coefficients (MFCCs)** for feature extraction, with CNNs offering a deep learning solution and GMMs providing a probabilistic approach.  

---

## GMM-Based Speaker Identification

The **GMM-based approach** models the speech characteristics of each speaker using Gaussian Mixture Models.  
### Features:
1. **Data Preprocessing**:  
   - Extracts **MFCC features** for each audio file.  
   - Pads or truncates features to ensure uniformity across samples.  
2. **Model Training**:  
   - Trains a separate GMM for each speaker using MFCC features.  
3. **Prediction**:  
   - Uses the log-likelihood of the GMMs to determine the most likely speaker.  
4. **Evaluation**:  
   - Calculates model accuracy on a test dataset.

---

## CNN-Based Speaker Identification

The **CNN-based approach** leverages deep learning to classify speakers.  
### Features:
1. **Data Preprocessing**:  
   - Similar MFCC feature extraction as the GMM model, but reshaped for CNN compatibility.  
2. **Data Augmentation**:  
   - Adds noise and applies time-stretching to improve model generalization.  
3. **Model Architecture**:  
   - Employs a multi-layer CNN with pooling, batch normalization, and dropout for efficient learning and regularization.  
4. **Training Enhancements**:  
   - Utilizes callbacks like **early stopping** and **learning rate scheduling**.  
5. **Prediction**:  
   - Saves the trained model for future predictions on new audio data.

---

## Dataset

The dataset contains audio recordings from **50 speakers**, each organized into separate folders. Audio files are processed to extract **MFCC features**, which act as robust descriptors of vocal characteristics. Data augmentation techniques are applied to improve the generalization of the CNN model. Refer this link for dataset[Dataset link](https://drive.google.com/drive/folders/1qb_NCloA8p7r7IgXBMji--6HELPwLL1J?usp=drive_link)

---

## Training and Testing

1. **GMM Approach**:  
   - Splits the MFCC features into training and testing sets.  
   - Trains individual GMMs for each speaker and evaluates them based on log-likelihoods of test samples.  

2. **CNN Approach**:  
   - Uses augmented data for training to ensure robust learning.  
   - Reserves 20% of the dataset for validation and testing.  

### Performance
- **GMM**: Outputs a **log-likelihood-based prediction** for test samples.  
- **CNN**: Achieves high accuracy with deep learning-based feature extraction and classification.  

---

## Usage

### For GMM:
1. **Training**:
   - Run the script to train GMMs for each speaker. Models are saved as `gmm_models.pkl` and `label_encoder.pkl`.  
2. **Prediction**:
   - Use the `predict_speaker` function with MFCC features to predict the speaker.  

### For CNN:
1. **Training**:
   - Train the CNN model using the provided script. The model is saved as `speaker_identification_cnn_model.h5`.
2. **Prediction**:
   - Use the `predict_speaker` function for speaker classification on new audio files.  

---

## Installation and Dependencies

Install the required libraries:

```bash
pip install numpy librosa tensorflow scikit-learn
```



## Results and Future Work

- **Results**:  
  - The GMM-based approach achieves satisfactory accuracy, demonstrating the effectiveness of probabilistic models for speaker identification.  
  - The CNN-based model delivers higher accuracy, especially with augmented data, and is scalable for larger datasets.  

This project highlights the power of both **machine learning** and **deep learning** techniques for speech processing and can be extended to real-world applications like **voice authentication systems** or **personalized assistants**.
