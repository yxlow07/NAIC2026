# Technical Write-Up: Diabetic Retinopathy Screening

## 1. Model Overview
We use a stacking ensemble of **ResNet-34** and **EfficientNet-B4**, whose 5-class softmax outputs (10 features total) are fed into a **Random Forest** meta-classifier (`class_weight="balanced"`) for the final prediction. ResNet-34 was chosen for its lightweight residual architecture and fast inference, while EfficientNet-B4 complements it with higher capacity via compound scaling. The Random Forest stacker was selected over simpler linear models for its robustness to overfitting and ability to capture non-linear interactions between the base models' probability outputs.

## 2. Preprocessing & Augmentation
All images undergo: **Adaptive Cropping** (black border removal), **Graham's method**, **RGB-CLAHE** **Green-channel CLAHE**, **Circular ROI masking**, then resized/center-cropped to **384×384**. Minority classes were oversampled offline via spatial flips to address severe class imbalance.

## 3. Training Setup
Base CNNs were fine-tuned with **AdamW** (lr=2e-4, wd=1e-5) and a **LambdaLR** scheduler (2-epoch warmup + cosine decay), for up to **20 epochs** with **early stopping** (patience=5) on validation Macro F1. **Cross-Entropy Loss** with class weights and `WeightedRandomSampler` were used to handle imbalance. Mixed precision (`autocast`) accelerated training.

Training is split into two different phases, pre-training and post-training finetuning. During pre-training, the model is trained on the provided dataset and using part of the dataset to validate. During post training, an external dataset is used to evaluate the performance and fine-tune over the whole training set by using a lower LR and only evaluating on the external dataset to prevent overfitting.

## 4. Validation Strategy
Stratified **80/10/10** train/val/test split. Model checkpoints selected by **Macro F1-score** (not accuracy) to ensure fair evaluation across all severity grades. The stratified split was performed before any augmentation to prevent data leakage. Online augmentations (flips, colour jitter) were applied strictly to the training set at runtime, ensuring the validation and test sets reflect real-world, unseen clinical data.

## 5. Efficiency
The inference pipeline uses only two lightweight CNNs plus a Random Forest stacker—no Vision Transformers or heavy tree-boosting models. ResNet-34 (~21M params) and EfficientNet-B4 (~19M params) together total ~40M parameters, well under the footprint of a single ViT-B/16 (~86M). On a T4 GPU, end-to-end inference (preprocessing + two forward passes + RF predict) runs at roughly **~17 images/sec** (batch size 16), making it suitable for clinical deployment. The entire flow is encapsulated in a single `infer.py` script.

## 6. Strategic Trade-offs & Defenses
- **Offline Data Augmentation (Data Overlap)**: To address the severe class imbalance and provide sufficient minority class samples for stable validation (using Macro F1), we implemented offline augmentations before the data split. We accepted the trade-off of having spatially augmented variants of the same image across splits to ensure minority classes had statistical representation during evaluation, and to simulate real-world scenarios where models must evaluate multiple, slightly varied images of returning patients.
- **Data-Centric AI vs. Hyperparameter Tuning**: Rather than executing exhaustive hyperparameter grid searches, we prioritized a Data-Centric approach. We invested our computational budget into building a robust preprocessing pipeline (Graham's method, CLAHE) and leveraged established transfer-learning default parameters for our pre-trained models. This was combined with dynamic learning rate scheduling and early stopping, relying on our meta-classifier to smooth out sub-optimalities without requiring microscopic tuning of individual base models.
- **Macro F1-Score over Raw Accuracy**: Given the extreme class imbalance (as evidenced by Class 0 heavily outnumbering severe Class 3), raw accuracy is a highly misleading metric; a model could achieve high accuracy simply by over-predicting the majority class. We specifically used Validation Macro F1-score for model checkpointing and early stopping. Macro F1 treats all classes equally, heavily penalizing the model if it ignores rare but clinically severe cases, ensuring our final model is genuinely capable of safely screening the entire disease spectrum.

## Poster
![Poster](poster.png)