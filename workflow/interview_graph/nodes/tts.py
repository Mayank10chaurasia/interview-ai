from kokoro import KPipeline
import sounddevice as sd
from workflow.states.interview import InterviewState
import speech_recognition as sr

tts_pipeline = KPipeline(lang_code="a")


def tts(state: InterviewState):

    question = state["question"]

    generator = tts_pipeline(
        question,
        voice="af_heart",
        speed=1.0,
    )

    for _, _, audio in generator:
        sd.play(audio, samplerate=24000)
        sd.wait()
  

    return {
    "question": question
}


