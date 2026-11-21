import os
import re
import json
import uuid
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google import genai
from google.genai import types

APP_DIR = Path(__file__).parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

app = FastAPI(title="AI Shorts Maker API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout)[-3000:])
    return result

def analyze_video(video_path: Path, target_seconds: int):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Server এ GEMINI_API_KEY সেট করা হয়নি।")
    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=str(video_path))
    try:
        for _ in range(60):
            current = client.files.get(name=uploaded.name)
            if getattr(current, "state", None) and str(current.state).upper().endswith("ACTIVE"):
                uploaded = current
                break
            if getattr(current, "state", None) and str(current.state).upper().endswith("FAILED"):
                raise RuntimeError("AI service ভিডিওটি প্রস্তুত করতে পারেনি।")
            time.sleep(2)

        prompt = f"""
Analyze the full video and identify the most engaging, self-contained moments
for a short social video. Return ONLY valid JSON:
{{"clips":[{{"start":0.0,"end":5.0,"reason":"brief reason"}}]}}
Choose several non-overlapping clips whose combined duration is about {target_seconds}
seconds. Timestamps must be within the video. Prefer clear action, emotion,
surprise, humor, or meaningful dialogue. Avoid blank/intro/outro sections.
Do not invent moments. Keep each clip at least 2 seconds.
"""
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        text = response.text or ""
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise RuntimeError("AI থেকে সঠিক clip তালিকা পাওয়া যায়নি।")
        data = json.loads(match.group(0))
        clips = data.get("clips", [])
        cleaned = []
        for clip in clips:
            start = max(0.0, float(clip["start"]))
            end = max(start, float(clip["end"]))
            if end - start >= 2:
                cleaned.append((start, end))
        if not cleaned:
            raise RuntimeError("AI কোনো ব্যবহারযোগ্য মুহূর্ত খুঁজে পায়নি।")
        return cleaned
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

def make_short(source: Path, clips, target_seconds: int, output: Path):
    work = Path(tempfile.mkdtemp(prefix="shorts_"))
    try:
        # Limit total clip time to requested duration.
        chosen = []
        remaining = float(target_seconds)
        for start, end in clips:
            length = min(end - start, remaining)
            if length >= 1.5:
                chosen.append((start, start + length))
                remaining -= length
            if remaining <= 0:
                break
        if not chosen:
            raise RuntimeError("নির্বাচিত clip-এর দৈর্ঘ্য যথেষ্ট নয়।")

        parts = []
        for i, (start, end) in enumerate(chosen):
            part = work / f"part_{i:02d}.mp4"
            run([
                "ffmpeg", "-y", "-ss", str(start), "-i", str(source),
                "-t", str(end-start),
                "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,setsar=1",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
                "-map", "0:v:0", "-map", "0:a:0?", "-shortest",
                "-movflags", "+faststart", str(part)
            ])
            parts.append(part)

        listing = work / "concat.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", "-movflags", "+faststart", str(output)
        ])
    finally:
        shutil.rmtree(work, ignore_errors=True)

@app.get("/health")
def health():
    return {"status": "ok", "service": "AI Shorts Maker"}

@app.post("/api/create-shorts")
async def create_shorts(
    file: UploadFile = File(...),
    duration: int = Form(20),
):
    if duration not in (15, 20, 30, 45):
        raise HTTPException(400, "Duration must be 15, 20, 30, or 45 seconds.")
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in (".mp4", ".mov", ".m4v", ".webm", ".mkv"):
        suffix = ".mp4"

    job_id = uuid.uuid4().hex
    source = OUTPUT_DIR / f"{job_id}_source{suffix}"
    output = OUTPUT_DIR / f"{job_id}_short.mp4"
    try:
        size = 0
        with source.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "ভিডিও সর্বোচ্চ ২০০ MB হতে পারবে।")
                f.write(chunk)

        if size == 0:
            raise HTTPException(400, "ভিডিও ফাইল খালি।")

        clips = analyze_video(source, duration)
        make_short(source, clips, duration, output)
        return FileResponse(
            output,
            media_type="video/mp4",
            filename="AI-Shorts-Maker.mp4",
            background=None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"ভিডিও তৈরি করা যায়নি: {str(exc)}")
    finally:
        try:
            source.unlink(missing_ok=True)
        except Exception:
            pass
