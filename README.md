# Invisible Watermarking for AI‑Generated Content

## Overview

Invisible Watermarking for AI‑Generated Content provides both **text** and **image** watermarking algorithms and a **web‑service framework** that:

- Embeds imperceptible watermarks into LLM outputs and images  
- Detects and verifies watermark presence to ensure provenance and integrity  
- Exposes a RESTful API and lightweight UI for on‑demand embedding/detection  

This project stems from the Yonsei University research proposal "Invisible Watermarking Algorithm and Service Development for Ensuring Content Reliability of Large Language Models (LLMs) and Image Generation Models"

---

## Features

| Module                   | Supported Content   | Algorithm Base                    | Detection Method              |
|--------------------------|---------------------|-----------------------------------|-------------------------------|
| **Text Watermarking**    | LLM‑generated text  | Green‑List / Red‑List Z‑Score     | Statistical Z‑score hypothesis test |
| **Image Watermarking**   | Synthetic images    | Watermark Anything Model (WAM)    | DBSCAN clustering on pixel shifts|
| **Web Service UI**       | Text & Image        | React + Tailwind + FastAPI        | Instant embed & detect panels  |

---

## Architecture

1. **Embedder**  
   - **Text**: Inserts subtle token‑probability bias during decoding  
   - **Image**: Projects localized watermark patterns via WAM pre‑ & post‑training  

2. **Extractor**  
   - **Text**: Computes green‑token ratio → Z‑score → Authenticity decision  
   - **Image**: Applies DBSCAN to extract clustered watermark regions  

3. **Web API & UI**  
   - FastAPI endpoints
   - React + Tailwind front‑end for drag‑and‑drop embedding and detection  
