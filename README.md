

<h1 align="center">Size-Zero</h1>
<p align="center">
  <img src="https://i.pinimg.com/originals/2a/ae/f6/2aaef61aa30a8b2ceb44d7f0778f2cdc.gif" 
       alt="Workout GIF"
       width="420">
</p>
<h3 align="center">Guiding Every Rep, Preventing Every Injury</h3>

<p align="center">
  <img src="https://img.shields.io/badge/AI%20Workout%20Tracker-Mediapipe%20%7C%20Streamlit%20%7C%20Gemini-blueviolet?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python" />
</p>

---

## Overview

**Size-Zero** is an AI-powered fitness assistant that uses computer vision to monitor workouts and ensure proper posture.  
With **MediaPipe**, it performs real-time pose estimation and rep counting. Post-workout analytics are generated using **Google Gemini** through **LangChain**, providing personalized performance summaries and improvement recommendations.  

---

## Tech Stack

| Component | Technology Used | Description |
|----------|------------------|-------------|
| Frontend | Streamlit | User interface for capturing workouts and displaying results |
| Computer Vision | MediaPipe | Real-time pose detection and body landmark tracking |
| AI/LLM Integration | Google Gemini (via LangChain) | Generates workout summaries and recommendations |
| Backend Logic | Python | Video processing, logic, and report generation |
| Environment | Anaconda / Google Colab / VS Code | Development and testing environments |

---

## Features

- Real-time pose detection using MediaPipe  
- Automatic rep counting and motion tracking  
- Post-session analytics and feedback using Gemini via LangChain  
- Streamlit-based clean user interface  
- Personalized performance summary and form improvement suggestions  

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/size-zero.git
cd size-zero

# Install dependencies
# IMPORTANT: Get your API key from Google AI Studio and add it to your environment variables.

# Run the application
streamlit run app.py
