import openai
import asyncio
import re
import whisper
import boto3
import pydub
from pydub import playback
from collections import deque
import speech_recognition as sr
from EdgeGPT import Chatbot, ConversationStyle

openai.api_key = "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"

recognizer = sr.Recognizer()
BING_WAKE_WORD = "bing"
GPT_WAKE_WORD = "gpt"
EXIT_WAKE_WORD = "exit"
bot = Chatbot(cookie_path='cookies.json')

model = None

def load_whisper_model():
  global model
  if model is None:
      # change 'small' to 'large' if you want a more accurate transcription
      model = whisper.load_model("small")
  return model

def get_wake_word(phrase):
  if BING_WAKE_WORD in phrase.lower():
    return BING_WAKE_WORD
  elif GPT_WAKE_WORD in phrase.lower():
    return GPT_WAKE_WORD
  elif EXIT_WAKE_WORD in phrase.lower():
    return EXIT_WAKE_WORD
  return None

def synthesize_speech(text, output_filename):
  session = boto3.Session(
    aws_access_key_id='AKIAXPWZOM7IPDNIJOOB',
    aws_secret_access_key='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',
    region_name='us-east-1'
  )
  polly = session.client('polly')
  response = polly.synthesize_speech(
    Text=text,
    OutputFormat='mp3',
    VoiceId='Matthew',
    Engine='neural'
  )

  with open(output_filename, 'wb') as f:
    f.write(response['AudioStream'].read())

def play_audio(file):
  sound = pydub.AudioSegment.from_file(file, format='mp3')
  playback.play(sound)

async def main():
  conversation_history = deque(maxlen=5)
  while True:
    with sr.Microphone() as source:
      recognizer.adjust_for_ambient_noise(source)
      print(f"Waiting for wake words 'ok bing', 'ok chatgpt', 'exit'...")

      while True:
        audio = recognizer.listen(source)

        try:
          with open("audio.wav", "wb") as f:
            f.write(audio.get_wav_data())

          model = load_whisper_model()
          result = model.transcribe("audio.wav")
          phrase = result["text"]

          print(f"You said: {phrase}")

          wake_word = get_wake_word(phrase)

          if wake_word is not None:
            break
          else:
            print("Not a wake word. Try again.")

        except Exception as e:
          print("Error transcribing audio: {0}".format(e))
          continue

      if wake_word == EXIT_WAKE_WORD:
        synthesize_speech("Thank you for allowing me to assist you. It was my pleasure to help.", 'thank_you.mp3')
        play_audio('thank_you.mp3')
        break

      print("Speak a prompt...")
      synthesize_speech('What can I help you with?', 'response.mp3')
      play_audio('response.mp3')
      audio = recognizer.listen(source)

      try:
        with open("audio_prompt.wav", "wb") as f:
          f.write(audio.get_wav_data())

        model = load_whisper_model()
        result = model.transcribe("audio_prompt.wav")
        user_input = result["text"]

        print(f"You said: {user_input}")

      except Exception as e:
        print("Error transcribing audio: {0}".format(e))
        continue

      if wake_word == BING_WAKE_WORD:
        response = await bot.ask(prompt=user_input, conversation_style=ConversationStyle.creative)

        for message in response["item"]["messages"]:
          if message["author"] == "bot":
            bot_response = message["text"]

        bot_response = re.sub('\[\^\d+\^\]', '', bot_response)

        print("Bot's response:", bot_response)
        synthesize_speech(bot_response, 'response.mp3')
        play_audio('response.mp3')
      else:
        while True:
          # Add the user's message to the conversation history
          conversation_history.append({"role": "user", "content": user_input})

          # Send the conversation history to the GPT-3.5-turbo API
          response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                    ] + list(conversation_history),
            temperature=0.5,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            stop=["\nUser:"],
          )

          bot_response = response["choices"][0]["message"]["content"]

          # Add the bot's response to the conversation history
          conversation_history.append({"role": "assistant", "content": bot_response})

          print("Bot's response:", bot_response)
          synthesize_speech(bot_response, 'response.mp3')
          play_audio('response.mp3')

          # Listen for user's follow-up question
          audio = recognizer.listen(source)

          try:
            with open("audio_prompt.wav", "wb") as f:
              f.write(audio.get_wav_data())

            model = load_whisper_model()
            result = model.transcribe("audio_prompt.wav")
            user_input = result["text"]

            print(f"You said: {user_input}")

            # Check if the user wants to end the conversation
            if "thank you" in user_input.lower():
              break

          except Exception as e:
            print("Error transcribing audio: {0}".format(e))
            continue

if __name__ == "__main__":
    asyncio.run(main())