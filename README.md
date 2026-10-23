# AgroVision: Analysis of Paddy Disease Classification and Monitoring System Using a Deep Learning Framework

Official implementation of the **AgroVision** research paper.

## 📌 Features
- **Spatial Visual Branch:** EfficientNet-B0 backbone for leaf lesion feature extraction.
- **Temporal Microclimate Branch:** 2-Layer stacked LSTM for preceding 7-day weather sequences (LWD, VPD, DTR, RH, Temp).
- **Cross-Attention Fusion:** Scaled Dot-Product Cross-Attention alignment between vision and agrometeorology.
- **Deployment Ready:** Quantized ONNX model served via FastAPI.

## 📊 Benchmark Results (Paddy Doctor Dataset)
| Architecture Variant | Accuracy | Macro F1-Score | Micro/Macro ROC-AUC |
| :--- | :---: | :---: | :---: |
| LSTM Only (Weather) | 35.98% | 0.168 | - |
| EfficientNet-B0 (Vision) | 93.42% | 0.929 | - |
| **Proposed Cross-Attention** | **95.44%** | **0.950** | **0.999** |

## 🚀 Quickstart

```bash
# Clone Repository
git clone [https://github.com/your-username/AgroVision-Paddy-Disease.git](https://github.com/your-username/AgroVision-Paddy-Disease.git)
cd AgroVision-Paddy-Disease

# Install Dependencies
pip install -r requirements.txt

# Train Pipeline
python train.py

# Export ONNX Model
python export_onnx.py

# Launch FastAPI Service
uvicorn app.main:app --reload
