from gtts import gTTS
import tempfile

def text_to_audio(text):

    fp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    tts = gTTS(
        text=text,
        lang="en"
    )

    tts.save(fp.name)

    return fp.name