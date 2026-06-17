def main():
    print("Hello from voice-ai-core!")


if __name__ == "__main__":
    main()

#
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import ollama

app = FastAPI(title="Voice AI Core - Token Streamer (8GB RAM Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def token_generator(user_prompt: str):
    """
    Asynchronously streams tokens from Llama 3.2 1B to maximize speed 
    and prevent memory spillover on an 8GB machine.
    """
    try:
        client = ollama.AsyncClient()
        
        response_stream = await client.chat(
            model='llama3.2:1b',  # Light footprint, fits entirely in RAM
            messages=[
                {
                    "role": "system", 
                    "content": "You are a concise, ultra-low latency real-time voice agent. Keep answers punchy."
                },
                {"role": "user", "content": user_prompt}
            ],
            stream=True
        )

        async for chunk in response_stream:
            token = chunk.get('message', {}).get('content', '')
            if token:
                yield f"data: {token}\n\n"
                await asyncio.sleep(0.01)

    except Exception as e:
        yield f"data: [ERROR: {str(e)}]\n\n"

@app.get("/stream")
async def stream_tokens(prompt: str):
    return EventSourceResponse(token_generator(prompt))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)