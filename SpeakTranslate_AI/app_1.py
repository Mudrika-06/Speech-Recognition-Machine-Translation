
import streamlit as st
import whisper
import torch
import tempfile
import os
import io
import csv
import shutil

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SpeakTranslate AI",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 SpeakTranslate AI")

st.subheader(
    "AI-Based Speech Recognition and Machine Translation"
)

st.write(
    "Convert spoken audio into written text and translate "
    "it into your preferred language using Artificial Intelligence."
)


# =========================================================
# 2. SESSION STATE
# =========================================================

DEFAULTS = {
    "recognized_text": "",
    "translated_text": "",
    "detected_language": "",
    "actual_source_language": "",
    "target_language": "",
    "feedback_records": [],
    "feedback_submitted": False
}

for key, value in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 3. LANGUAGE CONFIGURATION
# =========================================================

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


LANGUAGE_TO_WHISPER = {
    language: code
    for code, language in WHISPER_TO_LANGUAGE.items()
}


# =========================================================
# 4. DEVICE CONFIGURATION
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.sidebar.markdown("## ⚙️ System Information")

st.sidebar.write(
    f"**Processing Device:** {DEVICE.upper()}"
)

if DEVICE == "cuda":

    st.sidebar.success("GPU acceleration enabled.")

else:

    st.sidebar.info(
        "CPU mode active. Processing may take longer."
    )


# =========================================================
# 5. CHECK FFMPEG
# =========================================================

def check_ffmpeg():

    return shutil.which("ffmpeg") is not None


# =========================================================
# 6. LOAD WHISPER MODEL
# =========================================================

@st.cache_resource
def load_whisper_model():

    return whisper.load_model(
        "small",
        device=DEVICE
    )


# =========================================================
# 7. LOAD NLLB TRANSLATION MODEL
# =========================================================

@st.cache_resource
def load_translation_model():

    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=(
            torch.float16 if DEVICE == "cuda"
            else torch.float32
        )
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# =========================================================
# 8. SPEECH RECOGNITION
# =========================================================

def speech_to_text(
    audio_bytes,
    file_extension,
    speech_mode="Automatic",
    manual_language=None
):

    if not check_ffmpeg():

        raise FileNotFoundError(
            "FFmpeg is not installed. Add ffmpeg to "
            "packages.txt in the root of your repository."
        )

    model = load_whisper_model()

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temporary_file:

            temporary_file.write(audio_bytes)

            temporary_path = temporary_file.name

        options = {
            "task": "transcribe",
            "fp16": DEVICE == "cuda",
            "verbose": False,
            "condition_on_previous_text": False
        }

        # Manual language selection
        if speech_mode == "Manual Selection":

            if manual_language not in LANGUAGE_TO_WHISPER:

                raise ValueError(
                    "The selected source language is not supported."
                )

            options["language"] = (
                LANGUAGE_TO_WHISPER[manual_language]
            )

        # Automatic language detection
        # No language parameter is passed to Whisper.

        result = model.transcribe(
            temporary_path,
            **options
        )

        recognized_text = result.get(
            "text", ""
        ).strip()

        detected_code = result.get(
            "language", ""
        )

        return recognized_text, detected_code

    finally:

        if (
            temporary_path
            and os.path.exists(temporary_path)
        ):

            os.remove(temporary_path)


# =========================================================
# 9. TEXT CHUNKING
# =========================================================

def split_text(text, max_characters=850):

    words = text.split()

    chunks = []
    current_chunk = ""

    for word in words:

        if len(current_chunk) + len(word) + 1 <= max_characters:

            current_chunk += word + " "

        else:

            if current_chunk.strip():

                chunks.append(
                    current_chunk.strip()
                )

            current_chunk = word + " "

    if current_chunk.strip():

        chunks.append(
            current_chunk.strip()
        )

    return chunks


# =========================================================
# 10. MACHINE TRANSLATION
# =========================================================

def translate_text(
    text,
    source_language,
    target_language
):

    if not text.strip():

        raise ValueError(
            "No recognized text is available for translation."
        )

    if source_language not in LANGUAGES:

        raise ValueError(
            f"Unsupported source language: {source_language}"
        )

    if target_language not in LANGUAGES:

        raise ValueError(
            f"Unsupported target language: {target_language}"
        )

    if source_language == target_language:

        return text

    tokenizer, model = load_translation_model()

    source_code = LANGUAGES[source_language]
    target_code = LANGUAGES[target_language]

    # Set source language
    tokenizer.src_lang = source_code

    # Get target language token
    target_token_id = tokenizer.convert_tokens_to_ids(
        target_code
    )

    if (
        target_token_id is None
        or target_token_id == tokenizer.unk_token_id
    ):

        raise ValueError(
            f"Invalid target language code: {target_code}"
        )

    chunks = split_text(text)

    translated_chunks = []

    for chunk in chunks:

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

        with torch.inference_mode():

            output_tokens = model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                max_new_tokens=512,
                num_beams=4
            )

        translated = tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True
        )[0].strip()

        if not translated:

            raise ValueError(
                "The translation model returned an empty result."
            )

        translated_chunks.append(translated)

    return " ".join(translated_chunks)


# =========================================================
# 11. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.divider()

page = st.sidebar.radio(
    "📌 Navigation",
    [
        "Home",
        "About Project",
        "How It Works",
        "User Feedback"
    ]
)


# =========================================================
# 12. HOME PAGE
# =========================================================

if page == "Home":

    st.markdown("## 🎤 Audio Input")

    st.write(
        "Upload an audio file or record your voice."
    )

    audio_source = st.radio(
        "Choose audio input method:",
        [
            "Upload Audio File",
            "Record Audio"
        ]
    )

    audio_file = None
    audio_extension = ".wav"

    # Audio upload
    if audio_source == "Upload Audio File":

        audio_file = st.file_uploader(
            "Upload your audio file",
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

            audio_extension = (
                os.path.splitext(audio_file.name)[1].lower()
                or ".wav"
            )

    # Microphone recording
    else:

        audio_file = st.audio_input(
            "🎙️ Record your voice"
        )

        audio_extension = ".wav"

    st.divider()

    # =====================================================
    # LANGUAGE SELECTION
    # =====================================================

    st.markdown("## 🌍 Select Translation Languages")

    # ONLY TWO OPTIONS
    source_mode = st.radio(
        "🎤 Select your speech type:",
        [
            "Automatic Detection",
            "Manual Selection"
        ],
        index=0
    )

    col1, col2 = st.columns(2)

    manual_source_language = None

    with col1:

        if source_mode == "Manual Selection":

            manual_source_language = st.selectbox(
                "Source Language",
                list(LANGUAGES.keys())
            )

        else:

            st.info(
                "🔍 Source language will be detected automatically."
            )

    with col2:

        target_language = st.selectbox(
            "Target Language",
            list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index("Korean")
        )

    st.divider()

    # =====================================================
    # TRANSCRIBE AND TRANSLATE BUTTON
    # =====================================================

    if st.button(
        "🚀 Transcribe and Translate",
        type="primary",
        use_container_width=True
    ):

        if audio_file is None:

            st.warning(
                "⚠️ Please upload or record an audio file first."
            )

        elif (
            source_mode == "Manual Selection"
            and manual_source_language == target_language
        ):

            st.warning(
                "Source and target languages must be different."
            )

        else:

            try:

                audio_bytes = audio_file.getvalue()

                # STEP 1: SPEECH RECOGNITION

                with st.spinner(
                    "🎧 Step 1/2: Recognizing speech..."
                ):

                    recognized_text, detected_code = speech_to_text(
                        audio_bytes,
                        audio_extension,
                        source_mode,
                        manual_source_language
                    )

                if not recognized_text:

                    st.error(
                        "No speech was recognized. "
                        "Please try another recording."
                    )

                else:

                    detected_language = (
                        WHISPER_TO_LANGUAGE.get(
                            detected_code,
                            "Unknown"
                        )
                    )

                    # Determine actual source language

                    if source_mode == "Manual Selection":

                        actual_source_language = (
                            manual_source_language
                        )

                    else:

                        actual_source_language = (
                            detected_language
                        )

                    # Check supported source language

                    if actual_source_language not in LANGUAGES:

                        st.error(
                            "The detected language is not supported "
                            "by this application's language list. "
                            "Please use Manual Selection."
                        )

                    else:

                        st.success(
                            "✅ Speech recognition completed!"
                        )

                        st.info(
                            f"🔍 Detected Language: "
                            f"{detected_language}"
                        )

                        # DISPLAY RECOGNIZED TEXT

                        st.markdown("### 📝 Recognized Speech")

                        st.text_area(
                            "Speech Recognition Output",
                            value=recognized_text,
                            height=130,
                            key="current_recognized_text"
                        )

                        # STEP 2: TRANSLATION

                        if actual_source_language == target_language:

                            translated_text = recognized_text

                            st.warning(
                                "Source and target languages are "
                                "the same. No translation is required."
                            )

                        else:

                            with st.spinner(
                                "🌐 Step 2/2: Translating text..."
                            ):

                                translated_text = translate_text(
                                    recognized_text,
                                    actual_source_language,
                                    target_language
                                )

                        # SAVE RESULTS

                        st.session_state.recognized_text = (
                            recognized_text
                        )

                        st.session_state.translated_text = (
                            translated_text
                        )

                        st.session_state.detected_language = (
                            detected_language
                        )

                        st.session_state.actual_source_language = (
                            actual_source_language
                        )

                        st.session_state.target_language = (
                            target_language
                        )

                        st.session_state.feedback_submitted = False

                        st.success(
                            "🎉 Translation completed successfully!"
                        )

            except Exception as error:

                st.error(
                    "An error occurred while processing the audio."
                )

                st.exception(error)

    # =====================================================
    # DISPLAY TRANSLATION RESULTS
    # =====================================================

    if st.session_state.translated_text:

        st.divider()

        st.markdown("## 🌐 Translation Result")

        st.write(
            f"**Detected Language:** "
            f"{st.session_state.detected_language}"
        )

        st.write(
            f"**Source:** "
            f"{st.session_state.actual_source_language}"
        )

        st.write(
            f"**Target:** "
            f"{st.session_state.target_language}"
        )

        st.markdown("### 📝 Recognized Text")

        st.text_area(
            "Speech Recognition Output",
            value=st.session_state.recognized_text,
            height=120,
            key="saved_recognized_text"
        )

        st.markdown("### 🌍 Translated Text")

        st.text_area(
            "Machine Translation Output",
            value=st.session_state.translated_text,
            height=160,
            key="saved_translated_text"
        )

        # DOWNLOAD TRANSLATION

        st.download_button(
            "⬇️ Download Translation",
            data=st.session_state.translated_text,
            file_name="translated_text.txt",
            mime="text/plain",
            use_container_width=True
        )

        # DOWNLOAD COMPLETE REPORT

        report = f"""
SPEAKTRANSLATE AI
Speech Recognition and Machine Translation System

Detected Language:
{st.session_state.detected_language}

Source Language:
{st.session_state.actual_source_language}

Target Language:
{st.session_state.target_language}

RECOGNIZED TEXT:
{st.session_state.recognized_text}

TRANSLATED TEXT:
{st.session_state.translated_text}
"""

        st.download_button(
            "📄 Download Complete Report",
            data=report,
            file_name="speech_translation_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        # =================================================
        # USER FEEDBACK
        # =================================================

        st.divider()

        st.markdown("## 💬 User Feedback")

        st.write(
            "Are you satisfied with the speech recognition "
            "and translation?"
        )

        if not st.session_state.feedback_submitted:

            with st.form("feedback_form"):

                satisfaction = st.radio(
                    "⭐ Rate Your Experience:",
                    [
                        "😍 Excellent",
                        "😊 Good",
                        "😐 Average",
                        "😕 Poor",
                        "😞 Very Poor"
                    ]
                )

                comment = st.text_area(
                    "📝 Your Suggestions (Optional)",
                    placeholder=(
                        "Tell us how we can improve "
                        "SpeakTranslate AI..."
                    )
                )

                submitted = st.form_submit_button(
                    "📩 Submit Feedback",
                    use_container_width=True
                )

                if submitted:

                    st.session_state.feedback_records.append(
                        {
                            "Satisfaction": satisfaction,
                            "Comment": comment,
                            "Source Language": (
                                st.session_state.actual_source_language
                            ),
                            "Target Language": (
                                st.session_state.target_language
                            )
                        }
                    )

                    st.session_state.feedback_submitted = True

                    st.rerun()

        else:

            st.success(
                "🎉 Thank you for your valuable feedback!"
            )

            st.info(
                "💙 Your opinion helps us improve "
                "SpeakTranslate AI. 🌍"
            )


# =========================================================
# 13. ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.title("ℹ️ About SpeakTranslate AI")

    st.write(
        """
        SpeakTranslate AI is an Artificial Intelligence-based
        multilingual application that combines Automatic Speech
        Recognition and Neural Machine Translation.

        It converts spoken audio into written text using
        OpenAI Whisper and translates recognized text using
        Meta NLLB-200.
        """
    )

    st.markdown("### 🚀 Main Features")

    st.markdown(
        """
        - 🎤 Audio upload and microphone recording.
        - 🗣️ Speech-to-text conversion.
        - 🔍 Automatic language detection.
        - ✍️ Manual source-language selection.
        - 🌍 Multilingual machine translation.
        - 📥 Downloadable translation reports.
        - 💬 Emoji-based user feedback.
        - 📊 Feedback dashboard.
        """
    )

    st.markdown("### 🛠️ Technologies Used")

    st.markdown(
        """
        | Technology | Purpose |
        |---|---|
        | Python | Core programming language |
        | Streamlit | User interface |
        | OpenAI Whisper | Speech recognition |
        | Meta NLLB-200 | Machine translation |
        | PyTorch | Deep learning framework |
        | Hugging Face | Model integration |
        | FFmpeg | Audio processing |
        | GitHub | Version control |
        """
    )


# =========================================================
# 14. HOW IT WORKS
# =========================================================

elif page == "How It Works":

    st.title("⚙️ How SpeakTranslate AI Works")

    st.markdown("### Stage 1: Audio Input")

    st.write(
        "The user uploads an audio file or records speech "
        "using the microphone."
    )

    st.markdown("### Stage 2: Speech Recognition")

    st.write(
        "OpenAI Whisper processes the audio, recognizes "
        "the spoken words and identifies the source language."
    )

    st.markdown("### Stage 3: Machine Translation")

    st.write(
        "NLLB-200 receives the recognized text along with "
        "the source and target language codes and generates "
        "the translated output."
    )

    st.markdown("### Stage 4: User Feedback")

    st.write(
        "Users can rate the conversion and share comments "
        "to provide feedback about the system."
    )

    st.markdown("### 🔄 System Workflow")

    st.code(
        """
        Audio Input
             |
             v
        Audio Processing
             |
             v
        Whisper Speech Recognition
             |
             v
        Automatic / Manual Language Selection
             |
             v
        Recognized Text
             |
             v
        NLLB-200 Machine Translation
             |
             v
        Translated Output
             |
             v
        User Feedback
        """,
        language="text"
    )


# =========================================================
# 15. USER FEEDBACK DASHBOARD
# =========================================================

elif page == "User Feedback":

    st.title("💬 User Feedback Dashboard")

    st.write(
        "View feedback submitted during the current session."
    )

    records = st.session_state.feedback_records

    if records:

        st.success(
            f"Total feedback responses: {len(records)}"
        )

        for index, record in enumerate(records, start=1):

            with st.expander(
                f"Feedback #{index} - {record['Satisfaction']}"
            ):

                st.write(
                    f"**Rating:** {record['Satisfaction']}"
                )

                st.write(
                    f"**Comment:** "
                    f"{record['Comment'] or 'No comment provided'}"
                )

                st.write(
                    f"**Source:** {record['Source Language']}"
                )

                st.write(
                    f"**Target:** {record['Target Language']}"
                )

        st.markdown("### 📊 Satisfaction Summary")

        rating_counts = {}

        for record in records:

            rating = record["Satisfaction"]

            rating_counts[rating] = (
                rating_counts.get(rating, 0) + 1
            )

        for rating, count in rating_counts.items():

            st.write(
                f"{rating}: **{count} response(s)**"
            )

        # EXPORT FEEDBACK

        csv_buffer = io.StringIO()

        fieldnames = [
            "Satisfaction",
            "Comment",
            "Source Language",
            "Target Language"
        ]

        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

        st.download_button(
            "📥 Download Feedback CSV",
            data=csv_buffer.getvalue(),
            file_name="speaktranslate_feedback.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.info(
            "No feedback has been submitted yet. "
            "Complete a conversion and submit feedback. 😊"
        )


# =========================================================
# 16. FOOTER
# =========================================================

st.divider()

st.caption(
    "🌐 SpeakTranslate AI | "
    "AI-Based Speech Recognition and Machine Translation System"
)
