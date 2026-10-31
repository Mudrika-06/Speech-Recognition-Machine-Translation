# SpeakTranslate AI

## Speech Recognition and Machine Translation System

SpeakTranslate AI is an AI-powered multilingual application that
converts spoken audio into text and translates the recognized text
into a selected target language.

The application uses OpenAI Whisper for Automatic Speech Recognition
and Meta NLLB-200 for Neural Machine Translation.

## Features

- Upload audio files.
- Record audio using a microphone.
- Convert speech into written text.
- Detect the spoken language automatically.
- Select the source language manually.
- Translate text into another language.
- Support for 30 selected languages.
- Download translated text.
- Download a complete translation report.

## Technologies Used

- Python
- Streamlit
- OpenAI Whisper
- Meta NLLB-200
- PyTorch
- Hugging Face Transformers
- Google Colab
- GitHub

## Project Structure

SpeakTranslate_AI/
|
|-- app.py
|-- requirements.txt
|-- packages.txt
|-- README.md
|-- SpeakTranslate_AI_Colab.ipynb

## Installation on a Local Computer

### Step 1: Install Python

Install Python 3.10 or a compatible version.

### Step 2: Open the project folder

Open a terminal inside the project directory.

### Step 3: Create a virtual environment

Windows:

    python -m venv venv

Activate it:

    venv\Scripts\activate

### Step 4: Install dependencies

    pip install -r requirements.txt

Install FFmpeg separately and ensure it is available in PATH.

### Step 5: Run the application

    streamlit run app.py

The application will open in your browser.

## Running on Google Colab

1. Open Google Colab.
2. Upload the project files or clone your GitHub repository.
3. Install FFmpeg.
4. Install the Python dependencies.
5. Run the Streamlit application.
6. Expose the application using a tunneling service.

## AI Models

### OpenAI Whisper

Whisper performs Automatic Speech Recognition (ASR).
It converts spoken audio into text and identifies the language.

### Meta NLLB-200

NLLB-200 is a Neural Machine Translation model.
It translates text between supported languages.

This application uses the distilled 600M model.

## Processing Workflow

1. User uploads or records audio.
2. Whisper processes the audio.
3. Speech is converted into text.
4. The source language is identified or selected.
5. NLLB-200 translates the recognized text.
6. The translated output is displayed.
7. The user can download the result.

## Important Notes

- The first model download can take time.
- GPU execution is recommended for faster processing.
- Translation quality can vary depending on the language,
  audio quality, accent, and context.
- The application currently presents 30 selected languages,
  not every language supported by NLLB-200.
- NLLB-200 distilled 600M is licensed under CC-BY-NC 4.0.
  Review the model licence before any commercial use.
- Review model terms and licences before distributing the project.

## Future Enhancements

- Text-to-speech output.
- Translation history.
- More language options.
- Translation quality evaluation.
- Real-time speech translation.
- User accounts and saved translations.

## Project Purpose

This project demonstrates the integration of Automatic Speech
Recognition and Neural Machine Translation in a single application.
