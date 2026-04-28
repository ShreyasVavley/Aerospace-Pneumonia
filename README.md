# AeroScan

End-to-End Pneumonia Detection System powered by PyTorch ResNet18.

## Project Structure

- `/ml`: Training scripts, dataset loaders, and model exports.
- `/api`: FastAPI backend for handling predictions and CORS.
- `/web`: React (Next.js) frontend with a beautiful glassmorphic Midnight Obsidian theme.

## Setup Instructions

### 1. Model Training
```bash
cd ml
pip install torch torchvision
python train.py
```
This will generate `pneumonia_model.pth` in the `ml` folder.

### 2. API Backend
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
API runs on `http://localhost:8000`. Test the endpoint at `http://localhost:8000/docs`.

### 3. Frontend Web Interface
```bash
cd web
npm run dev
```
Access the application at `http://localhost:3000`.

## Features
- PyTorch ResNet18 Transfer Learning
- FastAPI Backend for High-Performance Inference
- Next.js Dashboard with Glassmorphism and Framer Motion animations
