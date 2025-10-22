# 🧠 SignBridge

SignBridge is an intelligent two-way translator that connects speech and sign language in real time, helping deaf and hearing individuals communicate naturally. This project involves two main pillars:

- **Speech → Text (STT)**: Converting spoken language to text using advanced speech recognition models
- **Sign → Text**: Recognizing sign language using computer vision and landmark-based techniques

## 🚀 Quick Start

### Running in Google Colab

1. Open any notebook from the `notebooks/` directory in Google Colab
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Clone this repository:
   ```
   !git clone https://github.com/nhahub/NHA-057.git
   %cd NHA-057
   ```
4. Run the setup script:
   ```
   !bash setup.sh
   ```
5. Define the data path:
   ```python
   DRIVE = '/content/drive/MyDrive/SignBridge_Data'
   ```

### Running Locally

#### On Linux/MacOS:
```bash
git clone https://github.com/nhahub/NHA-057.git
cd NHA-057
bash setup.sh
```

#### On Windows:
```cmd
git clone https://github.com/nhahub/NHA-057.git
cd NHA-057
setup.bat
```

## 📁 Project Structure

```
SignBridge/
├── README.md
├── requirements.txt
├── setup.sh
├── notebooks/
│   ├── 01_speech_to_text.ipynb
│   ├── 02_text_processing.ipynb
│   ├── 03_sign_landmarks.ipynb
│   ├── 04_train_sign_model.ipynb
│   └── 05_integration_demo.ipynb
├── ml/
│   ├── data_prep/
│   │   ├── extract_subset.py
│   │   ├── extract_landmarks.py
│   │   ├── preprocess_utils.py
│   │   └── visualize_landmarks.py
│   ├── models/
│   │   ├── sign_classifier.py
│   │   ├── smoothing.py
│   │   └── seq2seq_translator.py
│   ├── training/
│   │   ├── train_sign_classifier.py
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   └── train_utils.py
│   ├── inference/
│   │   ├── serve.py
│   │   ├── realtime.py
│   │   └── predictor.py
│   └── export/
│       ├── export_mlflow.py
│       ├── export_onnx.py
│       └── save_to_drive.py
├── nlp/
│   ├── stt/
│   │   ├── whisper_service.py
│   │   └── vosk_service.py
│   ├── tts/
│   │   ├── azure_tts.py
│   │   └── open_tts.py
│   ├── translation/
│   │   ├── seq2seq_model.py
│   │   └── postprocessing.py
│   └── utils/
│       └── language_utils.py
├── infra/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── github-actions.yml
├── mobile_integration/
│   ├── flutter_service.dart
│   └── webrtc_guide.md
├── configs/
│   ├── data_config.json
│   └── train_config.json
├── tests/
│   ├── test_stt.py
│   ├── test_data_prep.py
│   ├── test_model_inference.py
│   └── test_api_endpoints.py
└── .github/
    └── workflows/
        └── ci.yml
```

## 📂 Google Drive Structure

```
/MyDrive/SignBridge_Data/
├── metadata/
│   ├── MSASL_train.json
│   ├── MSASL_val.json
│   └── MSASL_classes.json
├── sign_landmarks/
│   ├── train/
│   ├── val/
│   └── test/
├── speech/
│   ├── raw_audio/
│   └── transcripts/
└── models/
    ├── sign/
    └── speech/
```

## 🧪 Development Workflow

1. Clone the repository
2. Create your branch: `git checkout -b <your-name>`
3. Work on your assigned tasks
4. Commit and push your changes: `git push origin <your-name>`
5. Create pull requests for reviewed and tested features

## 📋 Features

- **Speech-to-Text**: Advanced speech recognition using Whisper/Vosk
- **Sign Language Recognition**: MediaPipe-based landmark extraction and classification
- **Text Post-processing**: Grammar correction and punctuation restoration
- **Real-time Translation**: Integration of speech and sign language processing
- **Optimized Models**: Multiple architectures (LSTM, BiLSTM, CNN-LSTM, Transformers)
- **API Integration**: FastAPI endpoints for serving models
- **Deployment Options**: ONNX export and Docker containerization

## 🛠️ Technologies

- Python 3.10+
- PyTorch / TensorFlow
- MediaPipe
- Whisper / Vosk
- OpenCV
- Transformers (HuggingFace)
- FastAPI
- MLflow / TensorBoard

## 👥 Contributing

1. Follow the branching workflow described above
2. Ensure code quality with proper documentation and tests
3. Adhere to PEP8 standards
4. Include metrics and visualizations for model evaluation
5. Update README.md with any necessary information

## 📊 Evaluation Metrics

- **STT**: Word Error Rate (WER), inference latency
- **Sign Recognition**: Accuracy, F1-score, confusion matrix, per-class recall
- **Overall System**: End-to-end translation accuracy, real-time performance

## 📚 Documentation

Detailed documentation is available in the notebooks:
- `01_speech_to_text.ipynb`: Speech recognition implementation
- `02_text_processing.ipynb`: Text post-processing techniques
- `03_sign_landmarks.ipynb`: Landmark extraction and visualization
- `04_train_sign_model.ipynb`: Model training and evaluation
- `05_integration_demo.ipynb`: End-to-end system demonstration

## 📄 License

MIT

## 🙏 Acknowledgements

- MS-ASL dataset for sign language data
- OpenAI's Whisper for speech recognition
- Google's MediaPipe for pose estimation
- HuggingFace for NLP models and transformers