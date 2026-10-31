import streamlit as st
import whisper
import torch
import tempfile
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SpeakTranslate AI",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 SpeakTranslate AI")
st.subheader("Speech Recognition and Multilingual Machine Translation")

st.write(
    "Convert spoken audio into text and translate it into another language "
    "using Artificial Intelligence."
)

# --------------------------------------------------
# LANGUAGE CONFIGURATION
# --------------------------------------------------

LANGUAGES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl",
    "Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Arabic": "arb_Arab",
    "Bengali": "ben_Beng",
    "Urdu": "urd_Arab",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru",
    "Malayalam": "mal_Mlym",
    "Dutch": "nld_Latn",
    "Turkish": "tur_Latn",
    "Polish": "pol_Latn",
    "Ukrainian": "ukr_Cyrl",
    "Swahili": "swh_Latn",
    "Thai": "tha_Thai",
    "Vietnamese": "vie_Latn",
    "Indonesian": "ind_Latn",
    "Persian": "pes_Arab",
    "Greek": "ell_Grek"
}

# Whisper language codes mapped to the languages above
WHISPER_TO_LANGUAGE = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "bn": "Bengali",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ml": "Malayalam",
    "nl": "Dutch",
    "tr": "Turkish",
    "pl": "Polish",
    "uk": "Ukrainian",
    "sw": "Swahili",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "fa": "Persian",
    "el": "Greek"
}

# --------------------------------------------------
# DEVICE CONFIGURATION
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.sidebar.header("System Information")
st.sidebar.write(f"**Processing device:** {DEVICE.upper()}")

if DEVICE == "cuda":
    st.sidebar.success("GPU acceleration is available.")
else:
    st.sidebar.info(
        "CPU mode is active. Processing may take longer. "
        "Google Colab GPU runtime is recommended."
    )

# --------------------------------------------------
# LOAD WHISPER MODEL
# --------------------------------------------------

@st.cache_resource
def load_whisper_model():
    """Load the Whisper speech recognition model."""
    return whisper.load_model("base", device=DEVICE)


# --------------------------------------------------
# LOAD TRANSLATION MODEL
# --------------------------------------------------

@st.cache_resource
def load_translation_model():
    """Load the NLLB multilingual translation model."""

    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# --------------------------------------------------
# SPEECH RECOGNITION FUNCTION
# --------------------------------------------------

def speech_to_text(audio_bytes, file_extension):
    """
    Convert audio into text using OpenAI Whisper.

    Parameters:
        audio_bytes: Uploaded or recorded audio bytes.
        file_extension: Original audio file extension.

    Returns:
        Recognized text and detected language code.
    """

    model = load_whisper_model()

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temporary_file:

            temporary_file.write(audio_bytes)
            temporary_path = temporary_file.name

        result = model.transcribe(
            temporary_path,
            task="transcribe",
            fp16=(DEVICE == "cuda")
        )

        recognized_text = result.get("text", "").strip()
        detected_language = result.get("language", "")

        return recognized_text, detected_language

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


# --------------------------------------------------
# TEXT CHUNKING FUNCTION
# --------------------------------------------------

def split_text(text, max_characters=900):
    """
    Divide long text into smaller chunks for translation.
    This helps avoid exceeding the model's input limit.
    """

    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:

        if len(current_chunk) + len(word) + 1 <= max_characters:
            current_chunk += word + " "

        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            current_chunk = word + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# --------------------------------------------------
# MACHINE TRANSLATION FUNCTION
# --------------------------------------------------

def translate_text(text, source_language, target_language):
    """
    Translate recognized text using the NLLB model.
    """

    tokenizer, model = load_translation_model()

    source_code = LANGUAGES[source_language]
    target_code = LANGUAGES[target_language]

    tokenizer.src_lang = source_code

    text_chunks = split_text(text)

    translated_chunks = []

    for chunk in text_chunks:

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        target_token_id = tokenizer.convert_tokens_to_ids(
            target_code
        )

        with torch.no_grad():

            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                max_length=512,
                num_beams=4
            )

        translated_text = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]

        translated_chunks.append(translated_text)

    return " ".join(translated_chunks)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "About Project",
        "How It Works"
    ]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.markdown("### 🎤 Upload or Record Audio")

    st.write(
        "Choose an audio file from your device or record your voice "
        "using the microphone."
    )

    audio_source = st.radio(
        "Select audio input method:",
        ["Upload Audio File", "Record Audio"]
    )

    audio_file = None
    audio_extension = ".wav"

    if audio_source == "Upload Audio File":

        audio_file = st.file_uploader(
            "Upload your audio",
            type=[
                "wav",
                "mp3",
                "m4a",
                "mp4",
                "mpeg",
                "mpga",
                "webm",
                "ogg",
                "flac"
            ]
        )

        if audio_file is not None:
            audio_extension = os.path.splitext(
                audio_file.name
            )[1].lower() or ".wav"

    else:

        audio_file = st.audio_input(
            "Record your voice"
        )

        audio_extension = ".wav"

    st.divider()

    st.markdown("### 🌍 Select Translation Languages")

    col1, col2 = st.columns(2)

    with col1:

        source_options = [
            "Automatic Detection"
        ] + list(LANGUAGES.keys())

        source_language = st.selectbox(
            "Source Language",
            source_options,
            index=0
        )

    with col2:

        target_language = st.selectbox(
            "Target Language",
            list(LANGUAGES.keys()),
            index=1
        )

    st.divider()

    # --------------------------------------------------
    # PROCESS AUDIO
    # --------------------------------------------------

    if st.button(
        "🚀 Transcribe and Translate",
        type="primary",
        use_container_width=True
    ):

        if audio_file is None:

            st.warning(
                "Please upload an audio file or record your voice first."
            )

        elif (
            source_language != "Automatic Detection"
            and source_language == target_language
        ):

            st.warning(
                "Please select two different languages."
            )

        else:

            try:

                audio_bytes = audio_file.getvalue()

                # Step 1: Speech Recognition
                with st.spinner(
                    "Step 1/2: Recognizing speech..."
                ):

                    recognized_text, detected_code = speech_to_text(
                        audio_bytes,
                        audio_extension
                    )

                if not recognized_text:

                    st.error(
                        "No speech could be recognized. "
                        "Please try another audio recording."
                    )

                    st.stop()

                detected_language = WHISPER_TO_LANGUAGE.get(
                    detected_code
                )

                if source_language == "Automatic Detection":

                    if detected_language in LANGUAGES:

                        actual_source_language = detected_language

                    else:

                        st.error(
                            "The detected language is not included "
                            "in this application's language list. "
                            "Please select the source language manually."
                        )

                        st.stop()

                else:

                    actual_source_language = source_language

                st.success("Speech recognition completed!")

                # Display recognized text
                st.markdown("### 📝 Recognized Speech")

                st.info(
                    f"**Detected language:** {detected_language or 'Unknown'}"
                )

                st.text_area(
                    "Recognized Text",
                    value=recognized_text,
                    height=150
                )

                # Step 2: Translation
                if actual_source_language == target_language:

                    st.warning(
                        "Source and target languages are the same. "
                        "The recognized text is shown without translation."
                    )

                    translated_text = recognized_text

                else:

                    with st.spinner(
                        "Step 2/2: Translating text..."
                    ):

                        translated_text = translate_text(
                            recognized_text,
                            actual_source_language,
                            target_language
                        )

                st.success("Translation completed!")

                st.markdown("### 🌐 Translated Output")

                st.text_area(
                    "Translated Text",
                    value=translated_text,
                    height=180
                )

                # Download translated text
                st.download_button(
                    label="⬇️ Download Translation",
                    data=translated_text,
                    file_name="translated_text.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                # Download complete report
                report = f"""
SPEAKTRANSLATE AI
Speech Recognition and Machine Translation System

Source Language: {actual_source_language}
Target Language: {target_language}

RECOGNIZED TEXT:
{recognized_text}

TRANSLATED TEXT:
{translated_text}
"""

                st.download_button(
                    label="⬇️ Download Complete Report",
                    data=report,
                    file_name="speech_translation_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            except Exception as error:

                st.error(
                    "An error occurred while processing the audio."
                )

                st.exception(error)


# --------------------------------------------------
# ABOUT PROJECT PAGE
# --------------------------------------------------

elif page == "About Project":

    st.title("About SpeakTranslate AI")

    st.write(
        """
        SpeakTranslate AI is a multilingual speech recognition and
        machine translation application developed using Python
        and Artificial Intelligence.

        The system converts spoken language into written text
        through Automatic Speech Recognition (ASR) and then
        translates the recognized text into a selected target
        language using Neural Machine Translation (NMT).
        """
    )

    st.markdown("### Main Features")

    st.markdown(
        """
        - Audio upload and microphone recording.
        - Automatic speech recognition.
        - Multilingual text translation.
        - Automatic source language detection.
        - Manual source language selection.
        - Downloadable translated output.
        - Downloadable complete translation report.
        """
    )

    st.markdown("### Technologies Used")

    st.markdown(
        """
        | Technology | Purpose |
        |---|---|
        | Python | Core programming language |
        | Streamlit | User interface |
        | OpenAI Whisper | Speech recognition |
        | Meta NLLB-200 | Machine translation |
        | PyTorch | Deep learning framework |
        | Hugging Face Transformers | Model integration |
        | Google Colab | Development environment |
        | GitHub | Version control |
        """
    )


# --------------------------------------------------
# HOW IT WORKS PAGE
# --------------------------------------------------

elif page == "How It Works":

    st.title("How SpeakTranslate AI Works")

    st.write(
        "The application follows a two-stage AI processing workflow."
    )

    st.markdown("### Stage 1: Speech Recognition")

    st.write(
        """
        The user provides an audio recording through the microphone
        or uploads an existing audio file. OpenAI Whisper processes
        the audio and converts spoken words into written text.
        It also identifies the language spoken in the recording.
        """
    )

    st.markdown("### Stage 2: Machine Translation")

    st.write(
        """
        The recognized text is passed to the NLLB-200 translation
        model. The model processes the source text and generates
        the equivalent text in the selected target language.
        """
    )

    st.markdown("### System Workflow")

    st.code(
        """
        User Audio Input
                |
                v
        Audio Preprocessing
                |
                v
        Whisper Speech Recognition
                |
                v
        Recognized Text
                |
                v
        Source Language Identification
                |
                v
        NLLB-200 Machine Translation
                |
                v
        Translated Text
                |
                v
        Display / Download Output
        """,
        language="text"
    )

    st.info(
        "The system uses separate AI models for speech recognition "
        "and machine translation."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SpeakTranslate AI | AI-Based Speech Recognition "
    "and Machine Translation System"
)
