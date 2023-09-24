class Config:
    # Paths
    train_data_path = "data/train"
    val__data_path = "data/valid"
    test_data_path = "data/test"
    data_dir = "data"
    confusion_matrix_save_path = "/reports/confusion_matrix.png"
    classification_report_path = "/reports/classification_report.txt"
    model_report = "/reports/models_results.txt"
    model_dir = "saved_models"

    # Model training configuration
    seed = 42
    epochs = 15
    input_size = (224, 224)
    weights = "imagenet"
    learning_rate = 0.001
