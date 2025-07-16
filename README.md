# Real-time Sentiment Analysis Application

A full-stack application for real-time sentiment analysis using React frontend and FastAPI backend with machine learning models.

## 🏗️ Architecture Overview

### High-Level Architecture

```
┌─────────────────┐    HTTP/REST    ┌─────────────────┐
│                 │    Requests     │                 │
│   React App     │◄───────────────►│   FastAPI       │
│   (Frontend)    │                 │   (Backend)     │
│   Port: 3000    │                 │   Port: 8000    │
└─────────────────┘                 └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │                 │
                                    │   ONNX Model    │
                                    │   (Inference)   │
                                    │                 │
                                    └─────────────────┘
```

### Detailed Component Architecture

#### Frontend (React + TypeScript + Vite)
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS with custom gradients
- **Features**:
  - Real-time text input analysis
  - Responsive design for mobile/desktop
  - Loading states and error handling
  - Character count and processing time display

#### Backend (FastAPI + Python)
- **Framework**: FastAPI with automatic OpenAPI docs
- **ML Framework**: Transformers + ONNX Runtime
- **Features**:
  - REST API endpoints for sentiment prediction
  - CORS enabled for cross-origin requests
  - Optimized ONNX model inference
  - Response time tracking

#### Machine Learning Pipeline
- **Base Model**: Pre-trained transformer model
- **Optimization**: ONNX format for faster inference (~10-15ms)
- **Training**: Fine-tuning capabilities with custom datasets
- **Monitoring**: Weights & Biases (wandb) integration

### Data Flow

```
User Input → React Component → API Call → FastAPI → ONNX Model → Response → UI Update
```

1. User types text in React textarea
2. On keyup event, text is sent to FastAPI backend
3. Backend tokenizes text and runs inference on ONNX model
4. Model returns sentiment label, confidence score, and processing time
5. Results are displayed in real-time on the frontend

## 🔧 Environment Setup

### Required Environment Variables

Create a `.env` file in the `backend` directory with the following variables:

```bash
WANDB_API_KEY=
WANDB_DIR=
```

### Getting Your Wandb API Key

1. Sign up at [wandb.ai](https://wandb.ai)
2. Go to your account settings
3. Find your API key in the "API Keys" section
4. Copy the key to your `.env` file

### Why Wandb?

Weights & Biases provides:
- **Experiment Tracking**: Monitor training metrics in real-time
- **Model Versioning**: Track different model versions and performance
- **Hyperparameter Tuning**: Compare different training configurations
- **Collaboration**: Share results with team members
- **Artifact Management**: Store and version datasets and models

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/NevroHelios/Sentiment-analysis-realtime.git
cd Sentiment-analysis-realtime
git lfs install # to install lfs
git lfs pull # to pull the model weights

# Build and start all services
docker compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Manual Setup

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
fastapi run main.py --port 8000
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
# Accessible at http://localhost:5173
```

## 🤖 Model Training

### Fine-tuning Your Own Model

```bash
cd backend/scripts

# Basic training
python finetune.py --data your_data.jsonl --epochs 10

# Advanced training with custom parameters
python finetune.py \
  --data custom_dataset.jsonl \
  --epochs 20 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --output_dir ../models/custom_model
```

### Data Format

Your training data should be in JSONL format:
```json
{"text": "I love this product!", "label": "1"}
{"text": "This is terrible", "label": "0"}
```

Labels: `0` = Negative, `1` = Positive

## 📁 Project Structure

```
sentiment-realtime/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── App.tsx          # Main React component
│   │   └── App.css          # Styling
│   ├── Dockerfile           # Frontend container config
│   └── package.json         # Dependencies
├── backend/                 # FastAPI backend
│   ├── scripts/             # Training and utility scripts
│   │   ├── finetune.py     # Model fine-tuning
│   │   ├── training.py     # Training utilities
│   │   └── save_model_as_onnx.py
│   ├── models/             # Model storage
│   │   ├── finetuned/      # Fine-tuned models
│   │   └── onnx/           # ONNX optimized models
│   ├── main.py             # FastAPI application
│   ├── inference.py        # Model inference logic
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Backend container config
├── .github/workflows/       # CI/CD pipelines
└── docker-compose.yml      # Multi-service orchestration
```

## 🔄 CI/CD Pipeline

The project includes GitHub Actions for:
- Automated Docker builds
- Container health checks
- Image tagging with timestamps
- Multi-environment testing

## 🎯 Performance Optimizations

- **ONNX Runtime**: 2-3x faster inference compared to PyTorch
- **Containerization**: Consistent deployment across environments
- **Caching**: Docker layer caching for faster builds
- **Responsive Design**: Optimized for various screen sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
- **Visual Feedback**: Loading states and smooth animations
- **Accessibility**: Proper labels and semantic HTML

## 🔍 Performance

- **Model Size**: Optimized ONNX model for fast inference
- **Response Time**: Typically <10ms for sentiment analysis
- **Memory Usage**: Efficient CPU-based inference
- **Scalability**: Stateless API design for horizontal scaling

## 📝 Development Notes

- Frontend uses Vite's HMR for fast development
- Backend includes CORS middleware for cross-origin requests
- Model conversion to ONNX format included in fine-tuning pipeline
- TypeScript ensures type safety across the frontend
- Tailwind CSS provides utility-first styling approach

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

