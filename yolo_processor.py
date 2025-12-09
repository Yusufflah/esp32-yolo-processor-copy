import os
import requests
import json
from supabase import create_client, Client
from datetime import datetime
import time
from PIL import Image
import io
import cv2
import numpy as np
from ultralytics import YOLO

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # Use service key for write access
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load YOLO model (adjust path as needed)
model = YOLO('yolov5nu.pt')  # You can use yolov8s.pt, yolov8m.pt, etc.

def download_image(image_url):
    """Download image from URL"""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def process_image_with_yolo(image):
    """Process image with YOLO and return annotated image and detection results"""
    try:
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run YOLO inference
        results = model(opencv_image)
        
        # Annotate image with detections
        annotated_image = results[0].plot()
        
        # Convert back to PIL Image
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        
        # Extract detection information
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'class': model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        return annotated_pil, detections
    except Exception as e:
        print(f"Error processing image with YOLO: {e}")
        return None, []

def upload_processed_image(image, filename):
    """Upload processed image to Supabase storage"""
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)
        
        # Upload to processed-images bucket
        uploaded_file = supabase.storage.from_("processed-images").upload(
            f"processed_{filename}", 
            img_byte_arr.getvalue(),
            {"content-type": "image/jpeg"}
        )
        
        # Get public URL
        public_url = supabase.storage.from_("processed-images").get_public_url(f"processed_{filename}")
        return public_url
    except Exception as e:
        print(f"Error uploading processed image: {e}")
        return None

def update_processing_status(record_id, status, processed_image_url=None, processing_result=None, error_message=None, processing_time=None):
    """Update processing status in database"""
    try:
        update_data = {
            "status": status,
            "processed": status == "completed",
            "updated_at": datetime.now().isoformat()
        }
        
        if processed_image_url:
            update_data["processed_image_url"] = processed_image_url
            
        if processing_result:
            update_data["processing_result"] = processing_result
            
        if error_message:
            update_data["error_message"] = error_message
            
        if processing_time:
            update_data["processing_time"] = processing_time
            
        if status == "completed":
            update_data["processed_at"] = datetime.now().isoformat()
        
        response = supabase.table("yolo_processing").update(update_data).eq("id", record_id).execute()
        
        if hasattr(response, 'error') and response.error:
            print(f"Error updating status: {response.error}")
            return False
        return True
    except Exception as e:
        print(f"Error updating processing status: {e}")
        return False

def process_pending_images():
    """Process all pending images in the database"""
    try:
        # Fetch pending records
        response = supabase.table("yolo_processing").select("*").eq("status", "pending").execute()
        
        if hasattr(response, 'error') and response.error:
            print(f"Error fetching pending images: {response.error}")
            return
        
        pending_records = response.data
        print(f"Found {len(pending_records)} pending images")
        
        for record in pending_records:
            record_id = record['id']
            filename = record['filename']
            original_url = record['original_image_url']
            
            print(f"Processing: {filename}")
            
            # Update status to processing
            update_processing_status(record_id, "processing")
            
            start_time = time.time()
            
            try:
                # Download original image
                original_image = download_image(original_url)
                if not original_image:
                    raise Exception("Failed to download image")
                
                # Process with YOLO
                processed_image, detections = process_image_with_yolo(original_image)
                if not processed_image:
                    raise Exception("YOLO processing failed")
                
                # Upload processed image
                processed_url = upload_processed_image(processed_image, filename)
                if not processed_url:
                    raise Exception("Failed to upload processed image")
                
                processing_time = time.time() - start_time
                
                # Update record with results
                success = update_processing_status(
                    record_id, 
                    "completed", 
                    processed_url, 
                    detections,
                    processing_time=processing_time
                )
                
                if success:
                    print(f"✓ Successfully processed {filename} in {processing_time:.2f}s")
                    print(f"  Detections: {len(detections)} objects")
                else:
                    print(f"✗ Failed to update database for {filename}")
                    
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = str(e)
                print(f"✗ Error processing {filename}: {error_msg}")
                update_processing_status(
                    record_id, 
                    "failed", 
                    error_message=error_msg,
                    processing_time=processing_time
                )
                
    except Exception as e:
        print(f"Error in process_pending_images: {e}")

if __name__ == "__main__":
    print("Starting YOLO Image Processor...")
    process_pending_images()
    print("Processing completed!")
