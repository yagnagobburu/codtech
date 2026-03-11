import speech_recognition as sr
import os

def transcribe_audio(file_path):
   
    recognizer = sr.Recognizer()

    # Verify the file path exists
    if not os.path.exists(file_path):
        return "[Error] Audio file not found. Please ensure the file is in the folder."

    try:
        # Open the audio file
        with sr.AudioFile(file_path) as source:
            print("Reading audio data...")
            audio_data = recognizer.record(source)
            
            # Using Google's free Web Speech API
            print("Transcribing... please wait.")
            text = recognizer.recognize_google(audio_data)
            return text
            
    except sr.UnknownValueError:
        return "[Error] The AI could not understand the audio. It might be unclear or noisy."
    except sr.RequestError as e:
        return f"[Error] Could not connect to the recognition service: {e}"
    except Exception as e:
        return f"[Error] An unexpected error occurred: {e}"

def main():
    print("="*50)
    print("SPEECH-TO-TEXT CONVERTER - TASK 2")
    print("="*50)
    
    print("\nInstructions: Ensure your audio file is in .wav format for best results.")
    audio_file = input("\nEnter the name of your audio file (e.g., sample.wav): ").strip()

    print("\nProcessing...")
    result = transcribe_audio(audio_file)

    print("\n" + "="*20 + " TRANSCRIPTION RESULT " + "="*20)
    print(result)
    print("="*61)

if __name__ == "__main__":
    main()