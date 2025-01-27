import whisper #https://github.com/openai/whisper
from faster_whisper import WhisperModel
import sys
'''
Options for live transcription:
https://github.com/dhruvyad/uttertype

'''

def transcriptionTest():
  model = WhisperModel('base.en')
  segments, info = model.transcribe("assets/test-transcription.mp3", beam_size=5)
  print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
  print("------------------------------------")
  for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
  print("------------------------------------")


# Note: whisper & faster_whisper apparently don't yet work with Python 3.13, I had to downgrade to 3.10 to get it running.


if __name__ == ("__main__"):
  transcriptionTest()
