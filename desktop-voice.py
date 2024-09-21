import threading
from queue import Queue
import time
import numpy as np
import io
import pyaudiowpatch as pyaudio
import groq
import wave
import os
import audioop
import traceback

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
SILENCE_THRESHOLD = 400
SILENCE_DURATION = 1.8
PRE_SPEECH_BUFFER_DURATION = 1.0
API_KEY = "groq_api_key"

audio = pyaudio.PyAudio()
conversation_history = []

def get_loopback_device():
    wasapi_info = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = audio.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    #print(f"Default speakers: {default_speakers['name']}")

    if not default_speakers.get("isLoopbackDevice", False):
       # print("Searching for loopback device...")
        for loopback in audio.get_loopback_device_info_generator():
        #    print(f"Found loopback device: {loopback['name']}")
            if default_speakers["name"] in loopback["name"]:
         #       print(f"Selected loopback device: {loopback['name']}")
                return loopback
        print("No matching loopback device found.")
    else:
        print("Default speakers are already a loopback device.")
    return default_speakers

def is_silence(mic_data, desktop_data):
    mic_rms = audioop.rms(mic_data, 2)
    desktop_rms = audioop.rms(desktop_data, 2)
    return mic_rms < SILENCE_THRESHOLD and desktop_rms < SILENCE_THRESHOLD

def listen_for_speech():
    loopback_device = get_loopback_device()
    desktop_rate = int(loopback_device['defaultSampleRate'])

    mic_stream = audio.open(format=FORMAT, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    desktop_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=desktop_rate,
                                input=True, frames_per_buffer=CHUNK, input_device_index=loopback_device['index'])

    print("Listening for speech...")
    pre_speech_buffer_mic = []
    pre_speech_buffer_desktop = []
    pre_speech_chunks = int(PRE_SPEECH_BUFFER_DURATION * RATE / CHUNK)

    while True:
        mic_data = mic_stream.read(CHUNK)
        desktop_data = desktop_stream.read(CHUNK)

        # Resample desktop audio if necessary
        if desktop_rate != RATE:
            desktop_data, _ = audioop.ratecv(desktop_data, 2, CHANNELS, desktop_rate, RATE, None)

        # Convert stereo to mono for desktop audio
        if CHANNELS == 2:
            desktop_data = audioop.tomono(desktop_data, 2, 0.5, 0.5)

        pre_speech_buffer_mic.append(mic_data)
        pre_speech_buffer_desktop.append(desktop_data)

        if len(pre_speech_buffer_mic) > pre_speech_chunks:
            pre_speech_buffer_mic.pop(0)
            pre_speech_buffer_desktop.pop(0)

        if not is_silence(mic_data, desktop_data):
            print("Speech detected, start recording...")
            mic_stream.stop_stream()
            desktop_stream.stop_stream()
            mic_stream.close()
            desktop_stream.close()
            return record_audio(pre_speech_buffer_mic, pre_speech_buffer_desktop, loopback_device)

def record_audio(pre_speech_buffer_mic, pre_speech_buffer_desktop, loopback_device):
    desktop_rate = int(loopback_device['defaultSampleRate'])

    mic_stream = audio.open(format=FORMAT, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    desktop_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=desktop_rate,
                                input=True, frames_per_buffer=CHUNK, input_device_index=loopback_device['index'])

    frames_mic = pre_speech_buffer_mic.copy()
    frames_desktop = pre_speech_buffer_desktop.copy()

    silent_chunks = 0
    max_chunks = int(10 * RATE / CHUNK)  # Maximum 10 seconds of recording
    chunks_recorded = 0

    while chunks_recorded < max_chunks:
        mic_data = mic_stream.read(CHUNK)
        desktop_data = desktop_stream.read(CHUNK)

        # Resample desktop audio if necessary
        if desktop_rate != RATE:
            desktop_data, _ = audioop.ratecv(desktop_data, 2, CHANNELS, desktop_rate, RATE, None)

        # Convert stereo to mono for desktop audio
        if CHANNELS == 2:
            desktop_data = audioop.tomono(desktop_data, 2, 0.5, 0.5)

        frames_mic.append(mic_data)
        frames_desktop.append(desktop_data)

        chunks_recorded += 1

        if is_silence(mic_data, desktop_data):
            silent_chunks += 1
        else:
            silent_chunks = 0
        if silent_chunks > int(RATE / CHUNK * SILENCE_DURATION):
            break

    mic_stream.stop_stream()
    desktop_stream.stop_stream()
    mic_stream.close()
    desktop_stream.close()

    # Mix microphone and desktop audio
    mixed_frames = []
    for mic, desktop in zip(frames_mic, frames_desktop):
        mic_audio = np.frombuffer(mic, dtype=np.int16)
        desktop_audio = np.frombuffer(desktop, dtype=np.int16)

        # Ensure both arrays have the same length
        min_length = min(len(mic_audio), len(desktop_audio))
        mic_audio = mic_audio[:min_length]
        desktop_audio = desktop_audio[:min_length]

        # Mix the audio with adjusted weights and normalization
        mixed_audio = (mic_audio * 0.5 + desktop_audio * 0.5).astype(np.float32)

        # Normalize the mixed audio to prevent clipping
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 32767:
            mixed_audio = (mixed_audio / max_val * 32767).astype(np.int16)
        else:
            mixed_audio = mixed_audio.astype(np.int16)

        mixed_frames.append(mixed_audio.tobytes())

    return save_audio_to_file(b''.join(mixed_frames), "temp_audio.wav")

def save_audio_to_file(audio_data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Save as mono
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
    print(f"Audio saved to {filename}")
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
        {"role": "system", "content": "You are an assistant."}
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

def process_audio(audio_file):
    if audio_file is not None and os.path.getsize(audio_file) > 0:
        transcription = transcribe_audio(audio_file, API_KEY)
        print(f"Transcription: {transcription}")
        response = generate_response(transcription, API_KEY)
        # The response is already printed in the generate_response function
        os.remove(audio_file)  # Clean up the temporary file
    else:
        print("Failed to process audio")

def process_audio_loop():
    while True:
        try:
            print("Waiting for speech...")
            audio_file = listen_for_speech()
            print(f"Audio file created: {audio_file}")
            process_audio(audio_file)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            print("Restarting the loop...")

if __name__ == "__main__":
    try:
        print("Starting the voice assistant...")
        process_audio_loop()
    except Exception as e:
        print(f"A critical error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        input("Press Enter to exit...")
