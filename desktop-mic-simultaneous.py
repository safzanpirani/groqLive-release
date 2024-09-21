import threading
import queue
import time
import numpy as np
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
SILENCE_THRESHOLD = 300
SILENCE_DURATION = 1
PRE_SPEECH_BUFFER_DURATION = 0.5
API_KEY = "groq_api_key"

audio = pyaudio.PyAudio()
audio_queue = queue.Queue()
is_recording = False

def get_loopback_device():
    wasapi_info = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = audio.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    print(f"Default speakers: {default_speakers['name']}")

    if not default_speakers.get("isLoopbackDevice", False):
        print("Searching for loopback device...")
        for loopback in audio.get_loopback_device_info_generator():
            print(f"Found loopback device: {loopback['name']}")
            if default_speakers["name"] in loopback["name"]:
                print(f"Selected loopback device: {loopback['name']}")
                return loopback
        print("No matching loopback device found.")
    else:
        print("Default speakers are already a loopback device.")
    return default_speakers

def is_silence(mic_data, desktop_data):
    mic_rms = audioop.rms(mic_data, 2)
    desktop_rms = audioop.rms(desktop_data, 2)
    return mic_rms < SILENCE_THRESHOLD and desktop_rms < SILENCE_THRESHOLD

def audio_callback(in_data, frame_count, time_info, status):
    if is_recording:
        audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def start_recording(loopback_device):
    global is_recording
    is_recording = True

    desktop_rate = int(loopback_device['defaultSampleRate'])

    mic_stream = audio.open(format=FORMAT, channels=1, rate=RATE, input=True,
                            frames_per_buffer=CHUNK, stream_callback=audio_callback)
    desktop_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=desktop_rate,
                                input=True, frames_per_buffer=CHUNK,
                                input_device_index=loopback_device['index'],
                                stream_callback=audio_callback)

    return mic_stream, desktop_stream

def stop_recording(mic_stream, desktop_stream):
    global is_recording
    is_recording = False

    mic_stream.stop_stream()
    desktop_stream.stop_stream()
    mic_stream.close()
    desktop_stream.close()

def listen_for_speech():
    loopback_device = get_loopback_device()
    mic_stream, desktop_stream = start_recording(loopback_device)

    print("Listening for speech...")
    silence_counter = 0
    speech_detected = False
    recorded_frames_mic = []
    recorded_frames_desktop = []

    while True:
        if not audio_queue.empty():
            mic_data = audio_queue.get()
            desktop_data = audio_queue.get()

            if not is_silence(mic_data, desktop_data):
                silence_counter = 0
                speech_detected = True
            else:
                silence_counter += 1

            if speech_detected:
                recorded_frames_mic.append(mic_data)
                recorded_frames_desktop.append(desktop_data)

            if speech_detected and silence_counter > int(RATE / CHUNK * SILENCE_DURATION):
                break

        time.sleep(0.01)  # Small sleep to prevent busy-waiting

    stop_recording(mic_stream, desktop_stream)

    # Mix microphone and desktop audio
    mixed_frames = mix_audio(recorded_frames_mic, recorded_frames_desktop)

    return save_audio_to_file(mixed_frames, "temp_audio.wav")

def mix_audio(frames_mic, frames_desktop):
    mixed_frames = []
    for mic, desktop in zip(frames_mic, frames_desktop):
        mic_audio = np.frombuffer(mic, dtype=np.int16)
        desktop_audio = np.frombuffer(desktop, dtype=np.int16)

        # Ensure both arrays have the same length
        min_length = min(len(mic_audio), len(desktop_audio))
        mic_audio = mic_audio[:min_length]
        desktop_audio = desktop_audio[:min_length]

        # Mix the audio with adjusted weights
        mixed_audio = (mic_audio * 0.7 + desktop_audio * 0.3).astype(np.int16)
        mixed_frames.append(mixed_audio.tobytes())

    return b''.join(mixed_frames)

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
    if not transcription:
        return "No transcription available."

    client = groq.Client(api_key=api_key)

    try:
        stream = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": transcription}
            ],
            stream=True  # Enable streaming
        )
        return stream
    except Exception as e:
        return f"Error in response generation: {str(e)}"

def process_stream(stream):
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end='', flush=True)
            full_response += content
    print()  # New line after the full response
    return full_response

def process_audio(audio_file):
    if audio_file is not None and os.path.getsize(audio_file) > 0:
        print(f"Audio file size: {os.path.getsize(audio_file)} bytes")
        transcription = transcribe_audio(audio_file, API_KEY)
        print(f"Transcription: {transcription}")
        response_stream = generate_response(transcription, API_KEY)
        if isinstance(response_stream, str):  # Error occurred
            print(f"AI Assistant Response: {response_stream}")
        else:
            print("AI Assistant Response:")
            full_response = process_stream(response_stream)
        os.remove(audio_file)  # Clean up the temporary file
    else:
        print("Failed to process audio: File is empty or does not exist")

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
        finally:
            # Ensure recording is stopped if an exception occurs
            global is_recording
            if is_recording:
                is_recording = False

if __name__ == "__main__":
    try:
        print("Starting the voice assistant...")
        process_audio_loop()
    except Exception as e:
        print(f"A critical error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        input("Press Enter to exit...")
