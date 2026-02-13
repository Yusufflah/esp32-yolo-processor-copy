import os
import requests
import json
from supabase import create_client, Client
from datetime import datetime, timedelta
import time
from PIL import Image, ExifTags  # Tambahan ExifTags untuk membaca metadata
import io
import cv2
import numpy as np
from ultralytics import YOLO

# ==================== TELEGRAM KONFIGURASI ====================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_PATH = 'best.pt'
model = YOLO(MODEL_PATH)

MAX_RETRY_COUNT = 3
RETRY_DELAY_HOURS = 1

# ==================== FUNGSI BANTU UNTUK TELEGRAM ====================
def get_image_capture_time(image: Image.Image, fallback_time: str = None) -> str:
    """
    Ekstrak waktu jepret dari EXIF gambar.
    Prioritaskan DateTimeOriginal (tag 36867) -> DateTime (tag 306) -> fallback.
    """
    try:
        exif = image._getexif()
        if exif:
            # DateTimeOriginal
            dt_original = exif.get(36867)
            if dt_original:
                return dt_original
            # DateTime
            dt = exif.get(306)
            if dt:
                return dt
    except Exception as e:
        print(f"Gagal membaca EXIF: {e}")

    # Fallback: gunakan parameter atau waktu sekarang
    if fallback_time:
        return fallback_time
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def send_telegram_photo(photo_bytes: io.BytesIO, caption: str) -> bool:
    """
    Kirim foto ke Telegram menggunakan bot.
    Mengembalikan True jika sukses, False jika gagal atau token tidak tersedia.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token/chat_id tidak dikonfigurasi, lewati pengiriman.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1024]}  # caption max 1024 karakter
    files = {"photo": ("annotated.jpg", photo_bytes.getvalue(), "image/jpeg")}

    try:
        response = requests.post(url, data=data, files=files, timeout=30)
        response.raise_for_status()
        print("✓ Gambar berhasil dikirim ke Telegram")
        return True
    except Exception as e:
        print(f"✗ Gagal mengirim ke Telegram: {e}")
        return False

# ==================== FUNGSI UTAMA (tidak banyak berubah) ====================
def download_image(image_url):
    # ... (tetap sama seperti kode asli)
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def process_image_with_yolo(image, target_class='person'):
    # ... (tetap sama)
    try:
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model(opencv_image)
        annotated_image = results[0].plot()
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
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
    # ... (tetap sama)
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)

        supabase.storage.from_("processed-images").upload(
            f"processed_{filename}",
            img_byte_arr.getvalue(),
            {"content-type": "image/jpeg"}
        )
        public_url = supabase.storage.from_("processed-images").get_public_url(f"processed_{filename}")
        return public_url
    except Exception as e:
        print(f"Error uploading processed image: {e}")
        return None

def update_processing_status(record_id, status, processed_image_url=None,
                           processing_result=None, error_message=None,
                           processing_time=None, retry_count=None, last_error=None):
    # ... (tetap sama)
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
    # ... (tetap sama)
    retry_count = record.get('retry_count', 0)
    last_error_time = record.get('updated_at')

    if retry_count >= MAX_RETRY_COUNT:
        print(f"  Skipping {record['filename']} - reached max retry count ({retry_count}/{MAX_RETRY_COUNT})")
        return False
    if not last_error_time:
        return True
    try:
        last_error_dt = datetime.fromisoformat(last_error_time.replace('Z', '+00:00'))
        time_since_error = datetime.now() - last_error_dt
        if time_since_error < timedelta(hours=RETRY_DELAY_HOURS):
            print(f"  Skipping {record['filename']} - retry delay not reached")
            return False
    except Exception as e:
        print(f"  Error parsing time for {record['filename']}: {e}")
    return True

def process_single_image(record, process_failed=False):
    """Process single image record, tambahkan pengiriman Telegram jika ada deteksi."""
    record_id = record['id']
    filename = record['filename']
    original_url = record['original_image_url']
    current_status = record['status']

    print(f"Processing {filename} (status: {current_status})")

    update_processing_status(record_id, "processing")
    start_time = time.time()

    try:
        # 1. Download gambar asli
        original_image = download_image(original_url)
        if not original_image:
            raise Exception("Failed to download image")

        # --- Ambil waktu jepret dari EXIF gambar asli (fallback ke created_at record) ---
        fallback = record.get('created_at', None)
        capture_time = get_image_capture_time(original_image, fallback)

        # 2. Proses dengan YOLO
        processed_image, detections = process_image_with_yolo(original_image, target_class='person')
        if not processed_image:
            raise Exception("YOLO processing failed")

        # 3. Upload gambar yang sudah dianotasi ke Supabase
        processed_url = upload_processed_image(processed_image, filename)
        if not processed_url:
            raise Exception("Failed to upload processed image")

        processing_time = time.time() - start_time

        # 4. Update database dengan hasil deteksi
        success = update_processing_status(
            record_id,
            "completed",
            processed_url,
            detections,
            processing_time=processing_time
        )
        if not success:
            print(f"✗ Failed to update database for {filename}")
            return False

        print(f"✓ Successfully processed {filename} in {processing_time:.2f}s")
        print(f"  Detections: {len(detections)} objects")

        # 5. =============== KIRIM KE TELEGRAM JIKA ADA DETEKSI ===============
        if detections:
            # Siapkan bytes gambar untuk dikirim (sama seperti saat upload)
            img_bytes = io.BytesIO()
            processed_image.save(img_bytes, format='JPEG', quality=85)
            img_bytes.seek(0)

            # Buat caption singkat
            classes = list(set(d['class'] for d in detections))
            caption = (f"🔍 Objek Terdeteksi\n"
                       f"📸 Waktu Kejadian: {capture_time}\n"
                       f"📦 Jumlah: {len(detections)} Orang\n"
                       # f"🏷️ Kelas: {', '.join(classes)}\n"
                       f"🆔 File: {filename}")
            send_telegram_photo(img_bytes, caption)
        else:
            print("  Tidak ada deteksi, lewati pengiriman Telegram.")

        return True

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        print(f"✗ Error processing {filename}: {error_msg}")

        current_retry_count = record.get('retry_count', 0)
        new_retry_count = current_retry_count + 1

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
    # ... (tetap sama)
    try:
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
    # ... (tetap sama)
    try:
        response = supabase.table("yolo_processing").select("*").eq("status", "failed").execute()
        if hasattr(response, 'error') and response.error:
            print(f"Error fetching failed images: {response.error}")
            return
        failed_records = response.data
        print(f"Found {len(failed_records)} failed images")
        records_to_retry = [r for r in failed_records if should_retry_failed_image(r)]
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
    # ... (tetap sama)
    print("\n" + "="*50)
    print("Processing PENDING images...")
    print("="*50)
    process_pending_images()
    print("\n" + "="*50)
    print("Processing FAILED images (with retry logic)...")
    print("="*50)
    retry_failed_images()

def cleanup_old_failures():
    # ... (tetap sama)
    try:
        cutoff_time = datetime.now() - timedelta(days=7)
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
            if retry_count >= MAX_RETRY_COUNT and updated_at:
                try:
                    updated_dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    if updated_dt < cutoff_time:
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
    print("YOLO Image Processor with Retry Logic + Telegram Notifications")
    print("="*50)
    print(f"Max retry count: {MAX_RETRY_COUNT}")
    print(f"Retry delay: {RETRY_DELAY_HOURS} hours")
    print(f"Telegram: {'Enabled' if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else 'Disabled'}")
    print("="*50)

    process_all_images()
    print("\n" + "="*50)
    print("Cleaning up old failed records...")
    print("="*50)
    cleanup_old_failures()
    print("\n" + "="*50)
    print("Processing completed!")
    print("="*50)
