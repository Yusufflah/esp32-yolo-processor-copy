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

# Load YOLO model - GANTI DENGAN MODEL PERSON ONLY
# Pilih salah satu:
# 1. Model custom single-class
# model = YOLO('yolov5_person.pt')  # Model custom hanya person
# 2. Atau model standar yang akan difilter
model = YOLO('yolov5nu.pt')  # Model standar, akan difilter nanti

def download_image(image_url):
    """Download image from URL"""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def process_image_with_yolo_single_class(image, target_class='person'):
    """
    Process image with YOLO and return annotated image and detection results
    Hanya untuk single class (person)
    """
    try:
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run YOLO inference
        results = model(opencv_image)
        
        # Filter hanya deteksi 'person' dan buat annotated image khusus
        detections = []
        
        # Buat copy dari gambar asli untuk di-annotate
        annotated_image = opencv_image.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)
                
                # Hanya proses jika classnya adalah 'person'
                if class_name == target_class:
                    # Extract bounding box
                    bbox = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Draw bounding box dengan warna hijau untuk person
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"Person: {confidence:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Tambahkan ke detections
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox
                    }
                    detections.append(detection)
        
        # Convert back to PIL Image
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        
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

def update_processing_status(record_id, status, processed_image_url=None, 
                           processing_result=None, error_message=None, 
                           processing_time=None, person_count=None):
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
            
        if person_count is not None:
            update_data["person_count"] = person_count
            
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

def process_pending_images_single_class():
    """Process all pending images in the database - khusus untuk single class"""
    try:
        # Fetch pending records
        response = supabase.table("yolo_processing").select("*").eq("status", "pending" "failed").execute()
        
        if hasattr(response, 'error') and response.error:
            print(f"Error fetching pending images: {response.error}")
            return
        
        pending_records = response.data
        print(f"Found {len(pending_records)} pending images")
        print(f"Processing only 'person' class detections...")
        
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
                
                # Process with YOLO - single class version
                processed_image, detections = process_image_with_yolo_single_class(original_image, target_class='person')
                if not processed_image:
                    raise Exception("YOLO processing failed")
                
                # Upload processed image
                processed_url = upload_processed_image(processed_image, filename)
                if not processed_url:
                    raise Exception("Failed to upload processed image")
                
                processing_time = time.time() - start_time
                person_count = len(detections)
                
                # Update record with results
                success = update_processing_status(
                    record_id, 
                    "completed", 
                    processed_url, 
                    detections,
                    processing_time=processing_time,
                    person_count=person_count
                )
                
                if success:
                    print(f"✓ Successfully processed {filename} in {processing_time:.2f}s")
                    print(f"  Persons detected: {person_count}")
                    
                    # Log detail detections jika ada
                    if detections:
                        for i, det in enumerate(detections, 1):
                            print(f"    Person {i}: Confidence {det['confidence']:.2%}")
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
    print("Starting Single-Class (Person) YOLO Image Processor...")
    process_pending_images_single_class()
    print("Processing completed!")
