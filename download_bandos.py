import os
import requests
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
OUTPUT_DIR = Path("page_scans/bandos_v1")
BASE_URL   = "https://repositorio.agn.gob.mx/api/images/image/{}"
START_ID   = 11945231
# Script will auto-stop when it hits a 404 or non-image response
MAX_PAGES  = 500          # safety ceiling — adjust if needed
# ───────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer":    "https://repositorio.agn.gob.mx/",
    "Accept":     "image/webp,image/apng,image/*,*/*;q=0.8",
}

session = requests.Session()
session.headers.update(HEADERS)

downloaded = 0
for page_num in range(MAX_PAGES):
    image_id = START_ID + page_num
    url      = BASE_URL.format(image_id)
    filename = OUTPUT_DIR / f"page_{page_num + 1:04d}_{image_id}.jpg"

    try:
        resp = session.get(url, stream=True, timeout=30)

        # Stop if we get a 404 or the response isn't an image
        if resp.status_code == 404:
            print(f"✅ Reached end of sequence at image ID {image_id} (404). Done.")
            break

        if resp.status_code != 200:
            print(f"⚠️  Unexpected status {resp.status_code} at ID {image_id}. Stopping.")
            break

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type:
            print(f"✅ Non-image response at ID {image_id} (Content-Type: {content_type}). Done.")
            break

        # Save the image
        with open(filename, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        downloaded += 1
        print(f"[{page_num + 1:>4}] Saved: {filename.name}  ({resp.headers.get('Content-Length', '?')} bytes)")

    except requests.RequestException as e:
        print(f"❌ Error on ID {image_id}: {e}")
        break

print(f"\n🎉 Finished! {downloaded} pages saved to '{OUTPUT_DIR}'")