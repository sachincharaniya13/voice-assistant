import pyttsx3
import speech_recognition as sr
from logger import CustomLogger
import threading



class SpeechModule:
    """
    Handles Speech-to-Text (STT) and Text-to-Speech (TTS) functionalities.
    """

    def __init__(self):
        """
        Initialize the SpeechModule with TTS engine and recognizer for STT.
        """
        self.logger = CustomLogger().get_logger()
        self.logger.info("Initializing SpeechModule.")

        # Initialize Text-to-Speech (TTS) engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 200)
        self.tts_thread = None
        self.stop_tts_flag = False

    def speak(self, text: str):
        """
        Convert text to speech in a separate thread to allow interruption.
        """
        def tts_task():
            self.logger.info(f"TTS Output: {text}")
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                self.logger.error(f"Error in TTS: {e}")

        # Stop any ongoing TTS before starting a new one
        self.stop_tts()

        # Start the TTS task in a thread
        self.tts_thread = threading.Thread(target=tts_task)
        self.tts_thread.start()

    def stop_tts(self):
        """
        Stop any ongoing TTS operation.
        """
        if self.tts_thread and self.tts_thread.is_alive():
            self.logger.info("Stopping TTS...")
            self.tts_engine.stop()
            self.tts_thread.join()
