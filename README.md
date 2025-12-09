# ESP32-CAM YOLO Processor

Automatic YOLO image processing for ESP32-CAM motion detection using GitHub Actions.

## Setup

1. **Supabase Configuration**
   - Create tables and storage buckets
   - Set environment variables in GitHub Secrets

2. **ESP32-CAM Setup**
   - Upload modified Arduino code
   - Configure WiFi and Supabase credentials

3. **Automatic Processing**
   - GitHub Actions runs every 5 minutes
   - Processes pending images with YOLOv8
   - Stores results in Supabase

## File Structure