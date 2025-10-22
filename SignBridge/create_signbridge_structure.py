import os

structure = [
    # data
    "data/raw",
    "data/processed",
    "data/subsets",
    # ml
    "ml/data_prep",
    "ml/models",
    "ml/training",
    "ml/inference",
    "ml/export",
    # nlp
    "nlp/stt",
    "nlp/tts",
    "nlp/translation",
    "nlp/utils",
    # infra
    "infra/azure",
    # mobile integration
    "mobile_integration",
    # mlflow + configs + tests
    "mlflow",
    "configs",
    "tests"
]

files = [
    "ml/data_prep/download_subset.py",
    "ml/data_prep/extract_landmarks.py",
    "ml/data_prep/preprocess_utils.py",
    "ml/data_prep/visualize_landmarks.py",
    "ml/models/sign_classifier.py",
    "ml/models/smoothing.py",
    "ml/models/seq2seq_translator.py",
    "ml/training/train_sign_classifier.py",
    "ml/training/dataset.py",
    "ml/training/evaluate.py",
    "ml/training/train_utils.py",
    "ml/inference/serve.py",
    "ml/inference/realtime.py",
    "ml/inference/predictor.py",
    "ml/inference/smoothing_runtime.py",
    "ml/export/export_mlflow.py",
    "ml/export/export_onnx.py",
    "ml/export/save_azure_blob.py",
    "nlp/stt/whisper_service.py",
    "nlp/stt/vosk_service.py",
    "nlp/tts/azure_tts.py",
    "nlp/tts/open_tts.py",
    "nlp/translation/seq2seq_model.py",
    "nlp/translation/tokenizer.py",
    "nlp/translation/postprocessing.py",
    "nlp/utils/language_utils.py",
    "infra/Dockerfile",
    "infra/docker-compose.yml",
    "infra/azure/create_rg.sh",
    "infra/azure/deploy_aci.sh",
    "infra/azure/upload_to_blob.sh",
    "infra/azure/setup_mlflow.sh",
    "infra/github-actions.yml",
    "mobile_integration/flutter_service.dart",
    "mobile_integration/webrtc_guide.md",
    "mobile_integration/integration_test_plan.md",
    "mlflow/docker-compose.yml",
    "mlflow/config.yaml",
    "mlflow/README.md",
    "configs/lstm_config.yaml",
    "configs/transformer_config.yaml",
    "configs/mlflow_config.yaml",
    "tests/test_data_prep.py",
    "tests/test_model_inference.py",
    "tests/test_api_endpoints.py",
    "requirements.txt",
    "README.md"
]

for folder in structure:
    os.makedirs(folder, exist_ok=True)

for file in files:
    with open(file, "w", encoding="utf-8") as f:
        f.write("# " + file.split("/")[-1] + "\n")

print("✅ SignBridge project structure created successfully!")
