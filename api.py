# https://github.com/openai/whisper

# Note: whisper apparently doesn't yet work with Python 3.13, I had to downgrade to 3.10 to get it running.
import whisper

if __name__ == ("__main__"):
  model = whisper.load_model("tiny")
  result = model.transcribe("assets/test-transcription.mp3")
  print(result["text"])
