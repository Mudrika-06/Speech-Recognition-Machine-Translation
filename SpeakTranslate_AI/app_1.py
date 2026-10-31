import streamlit as st
import whisper
import torch
import tempfile
import os
import csv
import io

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="SpeakTranslate AI",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 SpeakTranslate AI")
st.subheader("AI-Based Speech Recognition and Machine Translation")

st.write(
    "Convert spoken audio into written text and translate it "
    "into another language using Artificial Intelligence."
)


# ==================================================
# 2. SESSION STATE INITIALIZATION
# ==================================================

if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""

if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""

if "actual_source_language" not in st.session_state:
    st.session_state.actual_source_language = ""

if "target_language" not in st.session_state:
    st.session_state.target_language = ""

if "feedback_records" not in st.session_state:
    st.session_state.feedback_records = []

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


# ==================================================
# 3. LANGUAGE CONFIGURATION
# ==================================================

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


# ==================================================
# 4. DEVICE CONFIGURATION
# ==================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.sidebar.header("⚙️ System Information")
st.sidebar.write(f"**Processing device:** {DEVICE.upper()}")

if DEVICE == "cuda":
    st.sidebar.success("GPU acceleration is available.")
else:
    st.sidebar.info(
        "CPU mode is active. Processing may take longer."
    )


# ==================================================
# 5. LOAD WHISPER MODEL
# ==================================================

@st.cache_resource
def load_whisper_model():

    model = whisper.load_model(
        "base",
        device=DEVICE
    )

    return model


# ==================================================
# 6. LOAD NLLB TRANSLATION MODEL
# ==================================================

@st.cache_resource
def load_translation_model():

    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=(
            torch.float16
            if DEVICE == "cuda"
            else torch.float32
        )
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ==================================================
# 7. SPEECH-TO-TEXT FUNCTION
# ==================================================

def speech_to_text(audio_bytes, file_extension):

    """
    Convert audio into text using OpenAI Whisper.

    Parameters:
        audio_bytes: Audio file content.
        file_extension: Original audio extension.

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

        recognized_text = result.get(
            "text",
            ""
        ).strip()

        detected_language = result.get(
            "language",
            ""
        )

        return recognized_text, detected_language

    finally:

        if (
            temporary_path
            and os.path.exists(temporary_path)
        ):

            os.remove(temporary_path)


# ==================================================
# 8. TEXT CHUNKING FUNCTION
# ==================================================

def split_text(text, max_characters=900):

    """
    Divide long text into smaller chunks
    before sending it to the translation model.
    """

    words = text.split()

    chunks = []
    current_chunk = ""

    for word in words:

        if (
            len(current_chunk) + len(word) + 1
            <= max_characters
        ):

            current_chunk += word + " "

        else:

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            current_chunk = word + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ==================================================
# 9. MACHINE TRANSLATION FUNCTION
# ==================================================

def translate_text(
    text,
    source_language,
    target_language
):

    """
    Translate recognized text using NLLB-200.
    """

    tokenizer, model = load_translation_model()

    source_code = LANGUAGES[source_language]
    target_code = LANGUAGES[target_language]

    tokenizer.src_lang = source_code

    text_chunks = split_text(text)

    translated_chunks = []

    target_token_id = tokenizer.convert_tokens_to_ids(
        target_code
    )

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

        with torch.inference_mode():

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

        translated_chunks.append(
            translated_text
        )

    return " ".join(translated_chunks)


# ==================================================
# 10. SIDEBAR NAVIGATION
# ==================================================

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


# ==================================================
# 11. HOME PAGE
# ==================================================

if page == "Home":

    st.markdown("## 🎤 Audio Input")

    st.write(
        "Upload an existing audio file or record your voice "
        "using the microphone."
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

    else:

        audio_file = st.audio_input(
            "🎙️ Record your voice"
        )

        audio_extension = ".wav"

    st.divider()

    # --------------------------------------------------
    # LANGUAGE SELECTION
    # --------------------------------------------------

    st.markdown("## 🌍 Language Selection")

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
    # TRANSCRIPTION AND TRANSLATION
    # --------------------------------------------------

    if st.button(
        "🚀 Transcribe and Translate",
        type="primary",
        use_container_width=True
    ):

        if audio_file is None:

            st.warning(
                "⚠️ Please upload an audio file "
                "or record your voice first."
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

                # STEP 1: SPEECH RECOGNITION

                with st.spinner(
                    "🎧 Step 1/2: Recognizing speech..."
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

                else:

                    detected_language = WHISPER_TO_LANGUAGE.get(
                        detected_code
                    )

                    # Determine source language

                    if source_language == "Automatic Detection":

                        if detected_language in LANGUAGES:

                            actual_source_language = (
                                detected_language
                            )

                        else:

                            st.error(
                                "The detected language is not "
                                "included in the available language "
                                "list. Please select the source "
                                "language manually."
                            )

                            actual_source_language = None

                    else:

                        actual_source_language = source_language

                    if actual_source_language is not None:

                        st.success(
                            "✅ Speech recognition completed!"
                        )

                        st.markdown(
                            "### 📝 Recognized Speech"
                        )

                        st.info(
                            f"Detected language: "
                            f"{detected_language or 'Unknown'}"
                        )

                        st.text_area(
                            "Recognized Text",
                            value=recognized_text,
                            height=150
                        )

                        # STEP 2: MACHINE TRANSLATION

                        if (
                            actual_source_language
                            == target_language
                        ):

                            translated_text = recognized_text

                            st.warning(
                                "Source and target languages "
                                "are the same. The recognized text "
                                "will be displayed without translation."
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

                        st.success(
                            "✅ Translation completed!"
                        )

                        # Save results in session state

                        st.session_state.recognized_text = (
                            recognized_text
                        )

                        st.session_state.translated_text = (
                            translated_text
                        )

                        st.session_state.actual_source_language = (
                            actual_source_language
                        )

                        st.session_state.target_language = (
                            target_language
                        )

                        # Reset feedback status for new conversion

                        st.session_state.feedback_submitted = False

            except Exception as error:

                st.error(
                    "An error occurred while processing the audio."
                )

                st.exception(error)

    # --------------------------------------------------
    # DISPLAY SAVED CONVERSION RESULTS
    # --------------------------------------------------

    if st.session_state.translated_text:

        st.divider()

        st.markdown("## 🌐 Translation Result")

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
            key="display_recognized_text"
        )

        st.markdown("### 🌍 Translated Text")

        st.text_area(
            "Machine Translation Output",
            value=st.session_state.translated_text,
            height=160,
            key="display_translated_text"
        )

        # Download translated text

        st.download_button(
            label="⬇️ Download Translation",
            data=st.session_state.translated_text,
            file_name="translated_text.txt",
            mime="text/plain",
            use_container_width=True
        )

        # Create complete report

        report = f"""
SPEAKTRANSLATE AI
Speech Recognition and Machine Translation System

Source Language:
{st.session_state.actual_source_language}

Target Language:
{st.session_state.target_language}

RECOGNIZED SPEECH:
{st.session_state.recognized_text}

TRANSLATED TEXT:
{st.session_state.translated_text}
"""

        st.download_button(
            label="📄 Download Complete Report",
            data=report,
            file_name="speech_translation_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        # --------------------------------------------------
        # USER FEEDBACK SECTION
        # --------------------------------------------------

        st.divider()

        st.markdown("## 💬 User Feedback")

        st.write(
            "We value your opinion! Tell us whether you are "
            "satisfied with the speech recognition and translation."
        )

        if not st.session_state.feedback_submitted:

            with st.form("user_feedback_form"):

                st.markdown(
                    "### ⭐ How satisfied are you with the conversion?"
                )

                satisfaction = st.radio(
                    "Choose your experience:",
                    [
                        "😍 Excellent",
                        "😊 Good",
                        "😐 Average",
                        "😕 Poor",
                        "😞 Very Poor"
                    ],
                    horizontal=True
                )

                st.markdown(
                    "### 📝 Share your experience"
                )

                feedback_comment = st.text_area(
                    "Suggestions or comments (Optional)",
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

                    feedback_record = {
                        "Satisfaction": satisfaction,
                        "Comment": feedback_comment,
                        "Source Language": (
                            st.session_state.actual_source_language
                        ),
                        "Target Language": (
                            st.session_state.target_language
                        )
                    }

                    st.session_state.feedback_records.append(
                        feedback_record
                    )

                    st.session_state.feedback_submitted = True

                    st.rerun()

        else:

            st.success(
                "🎉 Thank you for your valuable feedback!"
            )

            st.info(
                "💙 Your opinion helps us improve "
                "SpeakTranslate AI. Thank you for using our system! 🌍"
            )


# ==================================================
# 12. ABOUT PROJECT PAGE
# ==================================================

elif page == "About Project":

    st.title("ℹ️ About SpeakTranslate AI")

    st.write(
        """
        SpeakTranslate AI is a multilingual speech recognition
        and machine translation application developed using
        Python and Artificial Intelligence.

        The system converts spoken audio into written text
        using Automatic Speech Recognition (ASR) and then
        translates the recognized text into a selected target
        language using Neural Machine Translation (NMT).
        """
    )

    st.markdown("### 🚀 Main Features")

    st.markdown(
        """
        - 🎤 Audio upload and microphone recording.
        - 🗣️ Automatic speech recognition.
        - 🌍 Multilingual text translation.
        - 🔍 Automatic source language detection.
        - ✍️ Manual source language selection.
        - 📥 Downloadable translated output.
        - 💬 Emoji-based user satisfaction feedback.
        - 📝 User comments and suggestions.
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
        | Hugging Face Transformers | AI model integration |
        | FFmpeg | Audio processing |
        | GitHub | Version control |
        """
    )


# ==================================================
# 13. HOW IT WORKS PAGE
# ==================================================

elif page == "How It Works":

    st.title("⚙️ How SpeakTranslate AI Works")

    st.write(
        "The application follows a two-stage AI processing workflow."
    )

    st.markdown("### Stage 1: Speech Recognition")

    st.write(
        """
        The user uploads an audio file or records their voice.
        OpenAI Whisper processes the audio and converts spoken
        words into written text. It also identifies the spoken
        language.
        """
    )

    st.markdown("### Stage 2: Machine Translation")

    st.write(
        """
        The recognized text is passed to the NLLB-200 model.
        The model processes the source text and generates
        its translated equivalent in the selected target language.
        """
    )

    st.markdown("### Stage 3: User Feedback")

    st.write(
        """
        After receiving the translated output, the user can
        provide feedback using emoji-based satisfaction ratings
        and optional comments.
        """
    )

    st.markdown("### 🔄 System Workflow")

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
        User Satisfaction Feedback
                |
                v
        Feedback Record
        """,
        language="text"
    )


# ==================================================
# 14. USER FEEDBACK PAGE
# ==================================================

elif page == "User Feedback":

    st.title("💬 User Feedback Dashboard")

    st.write(
        "View feedback submitted during the current application session."
    )

    if st.session_state.feedback_records:

        st.success(
            f"Total feedback responses: "
            f"{len(st.session_state.feedback_records)}"
        )

        # Display feedback records

        for index, record in enumerate(
            st.session_state.feedback_records,
            start=1
        ):

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
                    f"**Source Language:** "
                    f"{record['Source Language']}"
                )

                st.write(
                    f"**Target Language:** "
                    f"{record['Target Language']}"
                )

        # Satisfaction summary

        st.markdown("### 📊 Satisfaction Summary")

        rating_counts = {}

        for record in st.session_state.feedback_records:

            rating = record["Satisfaction"]

            rating_counts[rating] = (
                rating_counts.get(rating, 0) + 1
            )

        for rating, count in rating_counts.items():

            st.write(
                f"{rating}: **{count} response(s)**"
            )

        # Download feedback as CSV

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

        writer.writerows(
            st.session_state.feedback_records
        )

        st.download_button(
            label="📥 Download Feedback as CSV",
            data=csv_buffer.getvalue(),
            file_name="speaktranslate_feedback.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.info(
            "No feedback has been submitted yet. "
            "Complete a conversion and submit your feedback "
            "to see it here. 😊"
        )


# ==================================================
# 15. FOOTER
# ==================================================

st.divider()

st.caption(
    "🌐 SpeakTranslate AI | "
    "AI-Based Speech Recognition and Machine Translation System"
)
