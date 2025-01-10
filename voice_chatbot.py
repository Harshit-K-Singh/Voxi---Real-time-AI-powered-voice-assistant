import webrtcvad
import pyaudio
import torch
import whisper
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ollama import Options
import ollama
from TTS.api import TTS
import simpleaudio as sa

# Initialize Whisper STT model
stt_model = whisper.load_model("base")

# Initialize Coqui TTS model
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC_ph", progress_bar=False)
tts_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Initialize WebRTC Voice Activity Detection (VAD)
vad = webrtcvad.Vad()
vad.set_mode(3)  # Most aggressive mode

# Audio capture parameters
RATE = 16000  # Sample rate
FRAME_DURATION_MS = 20  # Frame duration in ms (10, 20, or 30)
CHUNK = int(RATE * FRAME_DURATION_MS / 1000)  # Frame size

# Function to check if a frame contains speech
def is_speech(frame, sample_rate):
    """Check if a frame contains speech."""
    return vad.is_speech(frame, sample_rate)

# Function to transcribe audio frames
def transcribe_audio_chunk(frames):
    """Transcribe audio frames using Whisper."""
    audio_data = b"".join(frames)
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    result = stt_model.transcribe(audio_array)
    return result["text"]

# Function to generate a response using Ollama
def generate_response_ollama(prompt, max_tokens=100):
    """Generate response using Ollama."""
    response = ollama.chat(
        model='phi',
        messages=[{'role': 'user', 'content': prompt}],
        options=Options(
            max_tokens=50,
            temperature=0.7,
            top_p=0.3,
            num_ctx=512
        )
    )

    return response['message']['content']

# Function to convert text response to speech and play it
def speak_response(response):
    """Convert AI response to speech using Coqui TTS."""
    try:
        audio = tts_model.tts(response)
        audio = np.array(audio)
        audio = (audio * 32767).astype(np.int16)  # Convert to 16-bit PCM
        wave_obj = sa.WaveObject(audio.tobytes(), num_channels=1, bytes_per_sample=2, sample_rate=22050)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error in TTS: {e}")

# Main function for real-time audio recording and interaction
def record_and_transcribe():
    """Record audio in real-time, transcribe, generate response, and play it back."""
    print("Listening...")
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    silence_duration = 0
    speaking = False

    try:
        while True:
            frame = stream.read(CHUNK, exception_on_overflow=False)
            if is_speech(frame, RATE):
                speaking = True
                silence_duration = 0
                frames.append(frame)
            elif speaking:
                silence_duration += CHUNK / RATE
                if silence_duration > 1.5:  # Stop after 1.5 seconds of silence
                    print("Transcribing...")
                    transcription = transcribe_audio_chunk(frames)
                    print(f"User: {transcription}")

                    print("AI: ", end="", flush=True)
                    response = generate_response_ollama(transcription)
                    print(response)

                    speak_response(response)

                    frames = []  # Reset frames
                    silence_duration = 0
                    speaking = False
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# Run the main loop
if __name__ == "__main__":
    record_and_transcribe()
