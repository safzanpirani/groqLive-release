import pyaudio
import wave
import numpy as np
import groq
import os
import traceback
import threading
import queue

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 300
SILENCE_DURATION = 1
PRE_SPEECH_BUFFER_DURATION = 0.5
API_KEY = "groq_api_key"

conversation_history = []

audio = pyaudio.PyAudio()
audio_queue = queue.Queue()

def is_silence(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    squared = audio_data.astype(np.float64)**2
    mean_squared = np.mean(squared)
    if mean_squared > 0:
        rms = np.sqrt(mean_squared)
    else:
        rms = 0
    return rms < SILENCE_THRESHOLD

def listen_and_record():
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening for speech...")
    pre_speech_buffer = []
    pre_speech_chunks = int(PRE_SPEECH_BUFFER_DURATION * RATE / CHUNK)

    frames = []
    silent_chunks = 0
    max_chunks = int(10 * RATE / CHUNK)  # Maximum 10 seconds of recording
    chunks_recorded = 0
    recording = False

    try:
        while True:
            data = stream.read(CHUNK)

            if not recording:
                pre_speech_buffer.append(data)

                if len(pre_speech_buffer) > pre_speech_chunks:
                    pre_speech_buffer.pop(0)

                if not is_silence(data):
                    #print("Speech detected, start recording...")
                    recording = True
                    frames = pre_speech_buffer.copy()
            else:
                frames.append(data)
                chunks_recorded += 1

                if is_silence(data):
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if silent_chunks > int(RATE / CHUNK * SILENCE_DURATION) or chunks_recorded >= max_chunks:
                    print("End of speech detected.")
                    audio_queue.put(b''.join(frames))
                    frames = []
                    chunks_recorded = 0
                    silent_chunks = 0
                    recording = False

    except KeyboardInterrupt:
        print("Stopping the recording...")
    finally:
        stream.stop_stream()
        stream.close()

def save_audio_to_file(audio_data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
    #print(f"Audio saved to {filename}")
    return filename

def transcribe_audio(audio_file, api_key):
    client = groq.Client(api_key=api_key)

    try:
        with open(audio_file, 'rb') as file:
            completion = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=file,
                response_format="text"
            )
        return completion
    except Exception as e:
        return f"Error in transcription: {str(e)}"

def generate_response(transcription, api_key):
    global conversation_history

    if not transcription:
        return "No transcription available."

    client = groq.Client(api_key=api_key)

    # Add the user's message to the conversation history
    conversation_history.append({"role": "user", "content": transcription})

    # Prepare the messages for the API call
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ] + conversation_history

    try:
        print("AI Assistant Response:", end=" ", flush=True)
        full_response = ""
        for chunk in client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=messages,
            stream=True,
        ):
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print()  # New line after the complete response

        # Add the AI's response to the conversation history
        conversation_history.append({"role": "assistant", "content": full_response})

        # Optionally, limit the history to prevent it from growing too large
        if len(conversation_history) > 10:  # Keep last 10 messages
            conversation_history = conversation_history[-10:]

        return full_response
    except Exception as e:
        return f"Error in response generation: {str(e)}"

def process_audio():
    while True:
        audio_data = audio_queue.get()
        audio_file = save_audio_to_file(audio_data, "temp_audio.wav")

        if audio_file is not None:
            transcription = transcribe_audio(audio_file, API_KEY)
            print(f"Transcription: {transcription}")
            response = generate_response(transcription, API_KEY)
            # Remove this line to avoid duplicate printing
            # print(f"AI Assistant Response: {response}")
            os.remove(audio_file)  # Clean up the temporary file
        else:
            print("Failed to process audio")

if __name__ == "__main__":
    try:
        print("Starting the voice assistant...")

        # Start the audio processing thread
        processing_thread = threading.Thread(target=process_audio, daemon=True)
        processing_thread.start()

        # Start listening and recording
        listen_and_record()
    except Exception as e:
        print(f"A critical error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
    finally:
        audio.terminate()
        input("Press Enter to exit...")
