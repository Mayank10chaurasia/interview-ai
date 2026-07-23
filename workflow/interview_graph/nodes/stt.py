import os
import queue
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

# ==========================
# CONFIG
# ==========================

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

print("Loading Whisper model... (first time may take a few minutes)")

model = WhisperModel(
    "small",            # tiny, base, small
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
    num_workers=1,
)

print("✅ Whisper model loaded.\n")


# ==========================
# RECORD AUDIO
# ==========================

def record_audio():
    """
    Press ENTER to start recording.
    Press ENTER again to stop recording.
    """

    input("\n🎤 Press ENTER to start recording...")

    print("🔴 Recording...")
    print("Press ENTER again to stop.\n")

    audio_queue = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)

        audio_queue.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
    )

    stream.start()

    input()

    stream.stop()
    stream.close()

    audio = []

    while not audio_queue.empty():
        audio.append(audio_queue.get())

    if not audio:
        return None

    audio = np.concatenate(audio, axis=0)

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )

    sf.write(
        temp_file.name,
        audio,
        SAMPLE_RATE,
    )

    return temp_file.name


# ==========================
# LOCAL WHISPER TRANSCRIBE
# ==========================

def transcribe(audio_path):
    """
    Transcribe audio using Faster-Whisper.
    """

    segments, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=2,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 300
        },
        condition_on_previous_text=False,
    )

    transcript = []
    scores = []

    for segment in segments:
        transcript.append(segment.text.strip())

        if hasattr(segment, "avg_logprob"):
            scores.append(segment.avg_logprob)

    text = " ".join(transcript).strip()

    confidence = None

    if scores:
        confidence = round(sum(scores) / len(scores), 3)

    return text, confidence, info.language


# ==========================
# MAIN STT FUNCTION
# ==========================

def speech_to_text(state):

    audio_path = record_audio()

    if audio_path is None:
        print("❌ No audio captured")

        return {
            "answer": "",
            "history": [{
                "question": state["question"],
                "answer": ""
            }]
        }

    print(f"\n📁 Saved: {audio_path}")

    try:

        text, confidence, language = transcribe(audio_path)

        print("\n==============================")
        print("📝 TRANSCRIPT")
        print("==============================")
        print(text)
        print("==============================")
        print("🌍 Language :", language)
        print("📊 Confidence:", confidence)
        print("==============================\n")

    except Exception as e:

        print("❌ Whisper Error:", e)

        text = ""

    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass

    return {
        "answer": text,
        "history": [{
            "question": state["question"],
            "answer": text,
        }],
    }


