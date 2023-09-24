# Pneumonia Detection
This project aims to classify `Chest X-Ray` images as either `Normal` or `Pneumonia` using transfer learning with CNN models from Keras Applications.


[link](https://www.kaggle.com/code/mahtabranjbar/pneumonia-detection-transfer-learning/notebook) to project notebook in kaggle

---- 
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone this repository to your local machine:

   ```sh
   git clone https://github.com/MahtabRanjbar/TumorDetectAI.git
   ```

2. Navigate to the project directory:

   ```sh
   cd Brain-tumor-detection-app

   ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. If you want to retrain the model, you can execute the following command:
    ```sh
    python src/main.py
    ```
    consider to first download the data and put in data folder as it is said [here](data/README.md)

2. for ease of use and not installing all neccessary packages and having no conflict, you can run the  [notebook](notebooks/pneumonia-detection-transfer-learning.ipynb) of project

## Dataset
The dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) is organized into 3 folders `(train, test, val)` and contains subfolders for each image category `(Pneumonia/Normal)`. There are 5,863 X-Ray images (JPEG) and 2 categories `(Pneumonia/Normal)`. Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.


 You can download the dataset from the Kaggle website and extract it to the `data` directory in this project.
 For more information on the dataset and its structure,  see [here](data/README.md)


---
## Model

The following pre-trained models were loaded and their top layers removed before adding a new classification head, implementing transfer learning:

- DenseNet121
- MobileNetV2
- EfficientNetB4
- InceptionResNetV2
- InceptionV3
- MobileNetV3Large
- ResNet101
- ResNet50
- VGG19
- Xception


The models are trained using the Adam optimizer, which is a popular choice for deep learning tasks. 

---
## Results
The ResNet50 model achieved the highest test accuracy of 91.67%, making it the best model for this classification task. It has been saved for future use.

A table comparing all models' validation accuracy, test accuracy, and training time is provided in [`model_report.txt`](reports/model_report.txt). Confusion matrices and classification reports give further insight into best model's predictions.

This allows exploring the effectiveness of different pre-trained CNN architectures with transfer learning for automatic pneumonia detection from chest X-rays.

### Here is the result for best model which is resnet50

| class | precision | recall | f1-score | support |
|-|-|-|-|-|
| NORMAL | 0.87 | 0.91 | 0.89 | 234 |
| PNEUMONIA | 0.94 | 0.92 | 0.93 | 390 |
| **accuracy** |  |  | **0.92** | **624** |
| **macro avg** | **0.91** | **0.92** | **0.91** | **624** |
| **weighted avg** | **0.92** | **0.92** | **0.92** | **624** |


For further evaluation metrics and details, please check the [reports](reports/) folder.

---
## Contributing
Contributions are always welcome! If you have any ideas or suggestions, please feel free to open an issue or a pull request.

---
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information

---
## Contact
If you have any questions or comments about this project, please feel free to contact me at mahtabranjbar93@gmail.com











