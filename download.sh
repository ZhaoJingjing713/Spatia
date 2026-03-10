SAVE_DIR=./model_weights/Wan2.2-TI2V-5B
pip install -U "huggingface_hub[cli]<1.0.0"
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir $SAVE_DIR
