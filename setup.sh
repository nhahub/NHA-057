#!/bin/bash

# Function to detect if running in Colab
is_colab() {
    python -c "import sys; print('google.colab' in sys.modules)" | grep -q "True"
    return $?
}

# Function to install dependencies
install_dependencies() {
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Install additional dependencies for Colab
    if is_colab; then
        echo "Installing Colab-specific dependencies..."
        pip install pyngrok
    fi
    
    # Install MediaPipe with Holistic model
    pip install mediapipe
    
    echo "Dependencies installed successfully!"
}

# Function to mount Google Drive if in Colab
mount_drive() {
    if is_colab; then
        echo "Mounting Google Drive..."
        python -c "from google.colab import drive; drive.mount('/content/drive')"
        
        # Create base directory structure if it doesn't exist
        BASE_DIR="/content/drive/MyDrive/SignBridge_Data"
        
        if [ ! -d "$BASE_DIR" ]; then
            echo "Creating SignBridge_Data directory structure in Google Drive..."
            mkdir -p "$BASE_DIR/metadata"
            mkdir -p "$BASE_DIR/sign_landmarks/train"
            mkdir -p "$BASE_DIR/sign_landmarks/val"
            mkdir -p "$BASE_DIR/sign_landmarks/test"
            mkdir -p "$BASE_DIR/speech/raw_audio"
            mkdir -p "$BASE_DIR/speech/transcripts"
            mkdir -p "$BASE_DIR/models/sign"
            mkdir -p "$BASE_DIR/models/speech"
            echo "Directory structure created successfully!"
        else
            echo "SignBridge_Data directory already exists."
        fi
        
        # Create a symbolic link for easy access
        ln -sf "$BASE_DIR" ./SignBridge_Data
    else
        echo "Not running in Colab. Skipping Google Drive mount."
    fi
}

# Main function
main() {
    echo "🧠 Setting up SignBridge environment..."
    
    # Create virtual environment (if not in Colab)
    if ! is_colab; then
        if [ ! -d "venv" ]; then
            echo "Creating virtual environment..."
            python -m venv venv
            
            # Activate virtual environment based on OS
            if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
                source venv/Scripts/activate
            else
                source venv/bin/activate
            fi
            echo "Virtual environment created and activated."
        else
            echo "Virtual environment already exists."
            # Activate virtual environment based on OS
            if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
                source venv/Scripts/activate
            else
                source venv/bin/activate
            fi
        fi
    fi
    
    # Install dependencies
    install_dependencies
    
    # Mount Google Drive if in Colab
    mount_drive
    
    echo "✅ SignBridge environment setup complete!"
}

# Run the main function
main