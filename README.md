---
title: AeroScan API
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---
# 🫁 AeroScan

End-to-End Pneumonia Detection System powered by a custom PyTorch ResNet18 model.

## 🚀 Live Deployment
- **Frontend (Primary — Cloudflare Workers):** [https://aerospace-pneumonia.shreyasvavley.workers.dev](https://aerospace-pneumonia.shreyasvavley.workers.dev)
- **Frontend (Mirror — Vercel):** [https://aerospace-pneumonia.vercel.app](https://aerospace-pneumonia.vercel.app)
- **Backend (AI Inference API):** [https://shreyasvavley-aeroscan-api.hf.space](https://shreyasvavley-aeroscan-api.hf.space)

## 🏗️ Tech Stack
- **Machine Learning:** PyTorch, ResNet18, Grad-CAM Explainability
- **Backend Engine:** FastAPI, Docker
- **Frontend Interface:** Next.js 16, TailwindCSS v4, Framer Motion, Glassmorphism UI
- **Cloud Infrastructure:** Hugging Face Spaces (Backend), Cloudflare Workers via OpenNext (Frontend), Vercel (Mirror)

## 📁 Project Structure

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
