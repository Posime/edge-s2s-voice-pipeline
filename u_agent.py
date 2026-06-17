import asyncio
import re
import numpy as np
import httpx
import pyaudio
import torch
from kokoro import KPipeline
from transformers import pipeline as hf_pipeline

# 1. Initialize All Local Models
print("🎙️ Loading Silero VAD Model...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(_, _, _, _, _) = utils

print("🔤 Loading Whisper ASR Model...")
asr_pipeline = hf_pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

print("🧠 Initializing Kokoro TTS Pipeline...")
tts_pipeline_en = KPipeline(lang_code='a')
tts_pipeline_fr = KPipeline(lang_code='f')

# Audio configurations
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000      # Mic rate for Whisper/VAD
TTS_RATE = 24000  # Kokoro standard sample rate
CHUNK = 512       

# Initialize PyAudio globally
p = pyaudio.PyAudio()

# Open a dedicated output stream for your speakers (Signed 16-bit PCM configuration)
speaker_stream = p.open(
    format=pyaudio.paInt16, 
    channels=1, 
    rate=TTS_RATE, 
    output=True
)

def stream_audio_to_speakers(text, language ="en"):
    """Generates audio tokens and converts float32 waveforms to standard int16 byte arrays."""
    if language == "fr":
        generator = tts_pipeline_fr(text, voice='ff_swiss', speed= 1.0)
    else:
        generator = tts_pipeline_en(text, voice='af_heart', speed=1.0)
        
    for _, _, audio in generator:
        if audio is not None:

            if torch.is_tensor(audio):
              audio_np = audio.cpu().numpy()
            else:
                audio_np = audio
            # Scale float32 (-1.0 to 1.0) into structural 16-bit Integers to avoid sound card dropouts
            audio_scaled = (audio_np * 32767).astype(np.int16)
            # Flush the raw memory buffer directly into the physical laptop speaker
            speaker_stream.write(audio_scaled.tobytes())

async def process_voice_and_reply(raw_audio_data):
    audio_np = np.frombuffer(b''.join(raw_audio_data), dtype=np.int16).astype(np.float32) / 32768.0
    
    print("\n[🔤 Transcribing...]")
    transcription = asr_pipeline(audio_np)["text"].strip()
    print(f"👉 You said: '{transcription}'")
    
    if not transcription:
        print("⚠️ Silent frame detected. Speak up!")
        return

    url = "http://127.0.0.1:8000/stream"
    sentence_buffer = ""

    print("\n[🧠 Llama Responding Live...]")
    try:
        async with httpx.AsyncClient() as client:
            # Fixed stream call using explicit timeout configurations
            async with client.stream("GET", url, params={"prompt": transcription}, timeout=60.0) as response:
                # FIX: Explicitly call aiter_lines() to prevent the '__aiter__' generator crash
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        token = line.replace("data: ", "").strip()
                        print(token, end=" ", flush=True)
                        sentence_buffer += token + " "

                        if re.search(r'[.,!?\n]', sentence_buffer):
                            clean_sentence = sentence_buffer.strip()
                            if len(clean_sentence) > 1:
                                # Offload audio conversion and speaker writing to a separate execution thread
                                await asyncio.to_thread(stream_audio_to_speakers, clean_sentence)
                            sentence_buffer = ""
    except Exception as e:
        print(f"\n❌ Streaming error: {e}")

async def main_loop():
    mic_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("\n🟢 VOICE AGENT FULLY LIVE. Speak now...")
    
    is_speaking = False
    silence_frames = 0
    SILENCE_TIMEOUT = 25 
    speech_buffer = []

    try:
        while True:
            data = mic_stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            tensor_chunk = torch.from_numpy(audio_chunk)
            
            with torch.no_grad():
                speech_prob = vad_model(tensor_chunk, RATE).item()

            if speech_prob > 0.4:
                if not is_speaking:
                    print("\n[🎙️ Listening...]")
                    is_speaking = True
                silence_frames = 0
                speech_buffer.append(data)
            else:
                if is_speaking:
                    silence_frames += 1
                    speech_buffer.append(data)
                    
                    if silence_frames > SILENCE_TIMEOUT:
                        print("[⏹️ Processing...]")
                        mic_stream.stop_stream()
                        
                        await process_voice_and_reply(speech_buffer)
                        
                        speech_buffer = []
                        is_speaking = False
                        silence_frames = 0
                        print("\n🟢 READY NEXT TURN. Speak when ready...")
                        mic_stream.start_stream()

            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down agent...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        speaker_stream.stop_stream()
        speaker_stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(main_loop())