# Natural Scene Classification (VGG16 Transfer Learning)

Classifies 150×150 photos into six scene types (**buildings, forest, glacier, mountain, sea, street**) by fine-tuning an ImageNet-pretrained VGG16 in TensorFlow/Keras.

<!-- After running, save the plots to results/ and uncomment:
![Confusion matrix](results/confusion_matrix.png)
-->

## Results

| Metric | Value |
|---|---|
| Test accuracy | _fill in after running_ |
| Macro F1 | _fill in after running_ |
| Micro-average ROC AUC | _fill in after running_ |

## Approach

1. **Exploratory analysis:** sample images per class, class balance, RGB channel distributions and the average image of each class.
2. **Data pipeline:** rescaling, shear/zoom/flip augmentation on the training split, and an 80/20 train/validation split with no augmentation on validation or test.
3. **Model:** VGG16 base (`include_top=False`) with block 5 unfrozen, then Dense 512 → Dropout 0.5 → Dense 256 → Dropout 0.5 → Softmax 6.
4. **Training:** Adam at 1e-5, early stopping (patience 5), ReduceLROnPlateau and best-model checkpointing, up to 50 epochs.
5. **Evaluation:** a classification report, a confusion matrix and one-vs-rest ROC curves with AUC for each class.

## Dataset

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) on Kaggle has about 14k training images, 3k test images and 7k unlabeled images.

```
data/
├── seg_train/seg_train/<class>/*.jpg
├── seg_test/seg_test/<class>/*.jpg
└── seg_pred/seg_pred/*.jpg
```

## Run

```bash
pip install -r requirements.txt
# download and unzip the dataset into ./data, or point DATA_DIR elsewhere
DATA_DIR=./data python main.py
```

A GPU is recommended. The best model is saved to `$DATA_DIR/best_model.keras`.

## Author

**Faizan Hossain** · [LinkedIn](https://www.linkedin.com/in/faizan-hossain/)
