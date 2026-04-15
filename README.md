# Human Face Analysis - Real-time Facial Analysis Dashboard 

A modern, high-performance web application designed to demonstrate multi-modal computer vision capabilities including gaze tracking, lip movement analysis, head positioning, and distance estimation. 

Built with **MediaPipe**, **OpenCV**, **FastAPI**, and **Vite**.

## Overview
This project transforms a set of computer vision python scripts into a stunning, production-ready Proof of Concept. The backend processes a live camera feed via OpenCV and MediaPipe, overlays the detection metrics, and serves an MJPEG stream over HTTP to a beautiful, glassmorphism-styled frontend UI.

## Quick Start

### 1. Set up the Environment (Virtual Environment)
It is highly recommended (especially on macOS) to use a virtual environment so you don't pollute your system Python.
```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Backend API
While your `venv` is activated, run the API server.
```bash
uvicorn api.main:app --reload
```
*The API will run on `http://localhost:8000`*

### 3. Start the Frontend Dashboard
In a new terminal window, navigate to the frontend directory and start the Vite server.
```bash
cd frontend
npm install
npm run dev
```
*The application should now be live on `http://localhost:5173`!*

## 📁 Project Architecture
- `core/`: The computer vision intelligence powered by MediaPipe. Contains independent analyzers for face, gaze, lips, and distance.
- `api/`: The FastAPI backend serving the `video_feed` endpoint.
- `frontend/`: A Vite-powered Vanilla JS application featuring a responsive, neon-infused dark mode dashboard.
- `notebooks/`: Jupyter notebooks used during the exploratory data analysis and prototyping phase.

---
*Created as a Proof of Concept for advanced computer vision deployments.*
