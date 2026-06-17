# Local Edge-Optimized Speech-to-Speech (S2S) Pipeline

A low-latency, localized voice intelligence prototype engineered to run entirely on resource-constrained consumer hardware (tested on an 8GB RAM laptop).

This project demonstrates a cascaded Speech-to-Speech system architecture, bypassing slow cloud API tokens and disk I/O bottlenecks by processing continuous audio token streams entirely within volatile system memory.

## Architectural Flow

1. **Audio Ingestion & VAD:** Captures local microphone frames via PyAudio and processes them through an ultra-lightweight **Silero VAD** model (50MB) to handle dynamic user endpointing.
2. **Speech Recognition (ASR):** Converts voice waveforms to text instantly using OpenAI's **Whisper Tiny** via the Hugging Face Transformers engine.
3. **Inference & Streaming Core:** Ships text queries to a local **FastAPI** server hosting **Llama 3.2 1B** via Ollama, streaming text tokens back using Server-Sent Events (SSE).
4. **Speech Synthesis (TTS):** Intercepts live token fragments using an asynchronous punctuation-aware text buffer, piping text fragments directly into an 82M-parameter **Kokoro TTS** model for real-time, zero-shot audio array synthesis.
5. **In-Memory Audio Playback:** Scales float32 model outputs into standard signed 16-bit PCM integer byte matrices, streaming audio live through local laptop speakers out of memory to achieve near-instant Time-to-First-Audio (TTFA).

## Local Setup

Ensure you have system-level audio dependencies (`espeak-ng`, `portaudio`) installed on your OS.

```bash
# Clone the repository
git clone <your-repo-url>
cd voice-ai-core

# Pull the lightweight model engine
ollama pull llama3.2:1b

# Run the project instantly using the uv package manager
uv run python main.py          # Terminal 1: Starts the streaming backend server
uv run python u_agent.py # Terminal 2: Activates the unified voice engine loop
```
