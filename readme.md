Some of these scripts may not include conversation history. Should be easy enough to add, sorry for the laziness.


### 1. voice.py

This script uses microphone input exclusively.

- **Input Source**: Microphone
- **Functionality**: Captures audio from the computer's microphone, processes it, and performs transcription using whisper-v3 and then sends it to llama-3.1-70b.

### 2. desktop-voice.py

This script captures system/desktop audio.

- **Input Source**: Desktop/System Audio
- **Functionality**: Records the audio output from your computer (e.g., music, video playback, system sounds), processes it, and performs transcription using whisper-v3 and then sends it to llama-3.1-70b.

### 3. desktop-mic-simultaneous.py

This script simultaneously captures both microphone input and desktop/system audio.

- **Input Sources**: Microphone and Desktop/System Audio
- **Functionality**: Records audio from both the microphone and the computer's audio output, mixes these inputs, processes the combined audio, and performs transcription using whisper-v3 and then sends it to llama-3.1-70b.
