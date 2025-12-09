import os
import requests
import json
from supabase import create_client, Client
from datetime import datetime, timedelta
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

MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
# Load YOLO model
model = YOLO(MODEL_URL)  # Model standar

# Konfigurasi retry
MAX_RETRY_COUNT = 3  # Maksimal percobaan ulang untuk failed images
RETRY_DELAY_HOURS = 1  # Delay sebelum retry (dalam jam)

def download_image(image_url):
    """Download image from URL"""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def process_image_with_yolo(image, target_class='person'):
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
        
        # Extract detection information - filter only target class
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                
                # Filter hanya class yang diinginkan (default: person)
                if target_class == 'all' or class_name == target_class:
                    detection = {
                        'class': class_name,
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

def update_processing_status(record_id, status, processed_image_url=None, 
                           processing_result=None, error_message=None, 
                           processing_time=None, retry_count=None, last_error=None):
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
            
        if retry_count is not None:
            update_data["retry_count"] = retry_count
            
        if last_error:
            update_data["last_error"] = last_error
            
        if status == "completed":
            update_data["processed_at"] = datetime.now().isoformat()
            # Reset retry count jika berhasil
            update_data["retry_count"] = 0
            update_data["last_error"] = None
        
        response = supabase.table("yolo_processing").update(update_data).eq("id", record_id).execute()
        
        if hasattr(response, 'error') and response.error:
            print(f"Error updating status: {response.error}")
            return False
        return True
    except Exception as e:
        print(f"Error updating processing status: {e}")
        return False

def should_retry_failed_image(record):
    """
    Cek apakah failed image harus di-retry
    Berdasarkan retry_count dan waktu terakhir error
    """
    retry_count = record.get('retry_count', 0)
    last_error_time = record.get('updated_at')
    
    # Jika sudah mencapai maksimal retry, tidak usah di-retry lagi
    if retry_count >= MAX_RETRY_COUNT:
        print(f"  Skipping {record['filename']} - reached max retry count ({retry_count}/{MAX_RETRY_COUNT})")
        return False
    
    # Jika belum ada waktu error, retry saja
    if not last_error_time:
        return True
    
    # Cek apakah sudah cukup waktu sejak error terakhir
    try:
        last_error_dt = datetime.fromisoformat(last_error_time.replace('Z', '+00:00'))
        time_since_error = datetime.now() - last_error_dt
        
        # Jika error terjadi kurang dari RETRY_DELAY_HOURS yang lalu, tunggu dulu
        if time_since_error < timedelta(hours=RETRY_DELAY_HOURS):
            print(f"  Skipping {record['filename']} - retry delay not reached")
            return False
    except Exception as e:
        print(f"  Error parsing time for {record['filename']}: {e}")
        # Jika parsing error, tetap coba retry
        pass
    
    return True

def process_single_image(record, process_failed=False):
    """Process single image record"""
    record_id = record['id']
    filename = record['filename']
    original_url = record['original_image_url']
    current_status = record['status']
    
    print(f"Processing {filename} (status: {current_status})")
    
    # Update status to processing
    update_processing_status(record_id, "processing")
    
    start_time = time.time()
    
    try:
        # Download original image
        original_image = download_image(original_url)
        if not original_image:
            raise Exception("Failed to download image")
        
        # Process with YOLO
        processed_image, detections = process_image_with_yolo(original_image, target_class='person')
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
            return True
        else:
            print(f"✗ Failed to update database for {filename}")
            return False
            
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        print(f"✗ Error processing {filename}: {error_msg}")
        
        # Increment retry count
        current_retry_count = record.get('retry_count', 0)
        new_retry_count = current_retry_count + 1
        
        # Update status to failed dengan retry count baru
        update_processing_status(
            record_id, 
            "failed", 
            error_message=error_msg,
            processing_time=processing_time,
            retry_count=new_retry_count,
            last_error=error_msg
        )
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
        
        success_count = 0
        for record in pending_records:
            if process_single_image(record):
                success_count += 1
        
        print(f"✓ Processed {success_count}/{len(pending_records)} pending images successfully")
        
    except Exception as e:
        print(f"Error in process_pending_images: {e}")

def retry_failed_images():
    """Retry processing failed images based on retry logic"""
    try:
        # Fetch failed records
        response = supabase.table("yolo_processing").select("*").eq("status", "failed").execute()
        
        if hasattr(response, 'error') and response.error:
            print(f"Error fetching failed images: {response.error}")
            return
        
        failed_records = response.data
        print(f"Found {len(failed_records)} failed images")
        
        # Filter yang perlu di-retry
        records_to_retry = []
        for record in failed_records:
            if should_retry_failed_image(record):
                records_to_retry.append(record)
            else:
                print(f"  Not retrying {record['filename']} - retry logic conditions not met")
        
        print(f"Will retry {len(records_to_retry)} failed images")
        
        success_count = 0
        for record in records_to_retry:
            print(f"\nRetrying failed image: {record['filename']}")
            print(f"  Previous error: {record.get('error_message', 'Unknown error')}")
            print(f"  Retry count: {record.get('retry_count', 0)}/{MAX_RETRY_COUNT}")
            
            if process_single_image(record, process_failed=True):
                success_count += 1
        
        print(f"\n✓ Retried {success_count}/{len(records_to_retry)} failed images successfully")
        
    except Exception as e:
        print(f"Error in retry_failed_images: {e}")

def process_all_images():
    """Process both pending and failed images"""
    print("\n" + "="*50)
    print("Processing PENDING images...")
    print("="*50)
    process_pending_images()
    
    print("\n" + "="*50)
    print("Processing FAILED images (with retry logic)...")
    print("="*50)
    retry_failed_images()

def cleanup_old_failures():
    """Cleanup very old failed records that exceeded max retry"""
    try:
        # Hapus records yang sudah melewati maksimal retry dan error sudah lama
        cutoff_time = datetime.now() - timedelta(days=7)  # 7 hari yang lalu
        cutoff_time_str = cutoff_time.isoformat()
        
        response = supabase.table("yolo_processing").select("*").eq("status", "failed").execute()
        
        if hasattr(response, 'error') and response.error:
            print(f"Error fetching records for cleanup: {response.error}")
            return
        
        records = response.data
        cleaned_count = 0
        
        for record in records:
            retry_count = record.get('retry_count', 0)
            updated_at = record.get('updated_at')
            
            # Jika sudah melewati maksimal retry DAN error sudah lebih dari 7 hari
            if retry_count >= MAX_RETRY_COUNT and updated_at:
                try:
                    updated_dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    if updated_dt < cutoff_time:
                        # Delete record (atau update status menjadi 'abandoned')
                        supabase.table("yolo_processing").delete().eq("id", record['id']).execute()
                        print(f"  Cleaned up old failed record: {record['filename']}")
                        cleaned_count += 1
                except Exception as e:
                    print(f"  Error parsing date for cleanup: {e}")
        
        print(f"✓ Cleaned up {cleaned_count} old failed records")
        
    except Exception as e:
        print(f"Error in cleanup_old_failures: {e}")

if __name__ == "__main__":
    print("="*50)
    print("YOLO Image Processor with Retry Logic")
    print("="*50)
    print(f"Max retry count: {MAX_RETRY_COUNT}")
    print(f"Retry delay: {RETRY_DELAY_HOURS} hours")
    print("="*50)
    
    # Pilihan: bisa dipilih salah satu atau semua
    process_all_images()  # Proses semua (pending + failed dengan retry)
    
    # Opsional: cleanup records yang sudah sangat lama
    print("\n" + "="*50)
    print("Cleaning up old failed records...")
    print("="*50)
    cleanup_old_failures()
    
    print("\n" + "="*50)
    print("Processing completed!")
    print("="*50)
