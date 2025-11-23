import whisper
import os
import yt_dlp
from pyannote.audio import Pipeline
import torch

# === CONFIGURATION ===
HF_TOKEN = ""  # 🔑 Replace with your HF token
WHISPER_MODEL = "base"
LANGUAGE = "en"  # Set to None for auto-detect

def download_audio(url, output_path="temp_audio"):
    """Download audio using yt-dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Use WAV for diarization (higher quality)
            'preferredquality': '192',
        }],
            'postprocessor_args': {
            'ffmpeg:audio': ['-ar', '16000', '-ac', '1']  # 16kHz, mono
        },
        'quiet': False,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path + ".wav"


def transcribe_with_diarization(audio_path, model_name="base", language=None, hf_token=None):
    """Transcribe with speaker diarization using Whisper + pyannote."""
    # Load Whisper model
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    
    # Transcribe with word-level timestamps
    print("Transcribing with word timestamps...")

    result = model.transcribe(
        audio_path,
        language=language,
        fp16=False,
        word_timestamps=True  # ← Must be True
    )

    # --- Fallback if no word timestamps ---
    if "words" not in result or len(result["words"]) == 0:
        print("Warning: Word-level timestamps not available. Falling back to segment-level transcription.")
        # Return as single speaker (no diarization possible)
        segments_text = result.get("text", "").strip()
        if not segments_text:
            return [("SPEAKER_UNKNOWN", "")]
        return [("SPEAKER_00", segments_text)]

    # Run full speaker diarization
    print("Running speaker diarization...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    diarization = diarization_pipeline(audio_path)

    # Align speakers with transcribed words
    print("Aligning speakers with transcription...")
    speaker_segments = []
    for segment in diarization.itertracks(yield_label=True):
        seg, _, speaker = segment
        speaker_segments.append((seg.start, seg.end, speaker))

    def assign_speaker_to_words(words, speaker_segments):
        word_speaker = []
        for word in words:
            w_start = word["start"]
            w_end = word["end"]
            w_mid = (w_start + w_end) / 2  # Use midpoint to assign speaker
            assigned_speaker = "SPEAKER_UNKNOWN"
            for s_start, s_end, speaker in speaker_segments:
                if s_start <= w_mid <= s_end:
                    assigned_speaker = speaker
                    break
            word_speaker.append((word["word"], assigned_speaker))
        return word_speaker
    
    # Group words into speaker turns
    word_speaker_list = assign_speaker_to_words(result["words"], speaker_segments)

    # Reconstruct utterances by speaker
    current_speaker = None
    current_utterance = []
    utterances = []

    for word, speaker in word_speaker_list:
        if speaker != current_speaker:
            if current_utterance:
                utterances.append((current_speaker, " ".join(current_utterance)))
            current_speaker = speaker
            current_utterance = [word.strip()]
        else:
            current_utterance.append(word.strip())
    
    if current_utterance:
        utterances.append((current_speaker, " ".join(current_utterance)))

    # After transcription
    if "words" not in result or len(result["words"]) == 0:
        raise ValueError(
            "Whisper did not return word-level timestamps. "
            "Ensure the audio contains clear, sufficiently long speech."
        )

    return utterances if utterances else [("SPEAKER_UNKNOWN", "")]
    # return utterances


def transcribe_youtube_video(url, model_name="base", language=None, output_file=None, hf_token=None):
    audio_path = None
    try:
        print("Downloading audio...")
        audio_path = download_audio(url, output_path="temp_audio")
        
        utterances = transcribe_with_diarization(
            audio_path, 
            model_name=model_name, 
            language=language, 
            hf_token=hf_token
        )
        
        # Format output
        transcription_lines = []
        for speaker, text in utterances:
            line = f"[{speaker}]: {text}"
            transcription_lines.append(line)
        transcription = "\n".join(transcription_lines)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            print(f"Transcription saved to: {output_file}")
        
        return transcription

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


# === MAIN ===
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=FPpPTp7FIHY"
    output_file = "transcription_with_speakers.txt"

    transcription = transcribe_youtube_video(
        url=youtube_url,
        model_name=WHISPER_MODEL,
        language=LANGUAGE,
        output_file=output_file,
        hf_token=HF_TOKEN  # 🔑 Must set this!
    )
    print("\nTranscription with speakers:\n")
    print(transcription)