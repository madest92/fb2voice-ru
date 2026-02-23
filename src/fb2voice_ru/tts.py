"""Text-to-speech synthesis and audiobook generation.

Uses Silero TTS neural network to convert text to speech
and create MP3 audiobooks with metadata.
"""

import logging
import os
import re
import threading
import warnings
from multiprocessing import Manager, Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torchaudio
from mutagen.id3 import APIC, ID3, TALB, TIT2, TPE1
from tqdm import tqdm

from fb2voice_ru.normalizer import Normalizer

from .fb2 import FB2Parser

# Setup logging
logger = logging.getLogger(__name__)

# Suppress PyTorch warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.package"
)
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=".*multi_acc_v3_package.*"
)


class Book2Voice:
    """Convert FB2 e-books to audiobooks using speech synthesis.

    Loads FB2 file, parses structure, generates MP3 audiobooks
    using Silero TTS neural model with multiple voices.
    """

    # ================== CONSTANTS ==================

    # Silero TTS model URL
    SILERO_MODEL_URL = 'https://models.silero.ai/models/tts/ru'
    SILERO_MODELS = {
        'v5_1_ru.pt': ['aidar', 'baya', 'eugene', 'kseniya', 'xenia'],
        'v5_cis_base_v1.pt': [
            "ru_aigul", "ru_albina", "ru_alexandr", "ru_alfia",
            "ru_alfia2", "ru_bogdan", "ru_dmitriy", "ru_ekaterina",
            "ru_vika", "ru_gamat", "ru_igor", "ru_karina",
            "ru_kejilgan", "ru_kermen", "ru_marat", "ru_miyau",
            "ru_nurgul", "ru_oksana", "ru_onaoy", "ru_ramilia",
            "ru_roman", "ru_safarhuja", "ru_saida", "ru_sibday",
            "ru_zara", "ru_zhadyra", "ru_zhazira", "ru_zinaida",
            "ru_eduard",
        ],
    }
    MODEL_PATH = torch.hub.get_dir()

    SENTENCE_RE = re.compile(r'(?<=[.!?…])\s+')

    # Audio parameters
    _resampler = None
    SAMPLE_TTS_RATE = 48000
    SAMPLE_AUDIO_RATE = 22050
    PAUSE_DIALOG = 0.6       # Pause for dialogue (seconds)
    PAUSE_PARAGRAPH = 0.8    # Pause between paragraphs
    PAUSE_CHAPTER = 2.0      # Pause between chapters
    MAX_LEN = 800            # Max text length for single synthesis

    # ================== INIT ==================

    def __init__(self, fb2_path: str) -> None:
        """Initialize Book2Voice converter."""
        self.path = fb2_path
        self._parser = FB2Parser(fb2_path)
        self._book = self._parser.get_book()

    def __getattr__(self, name):
        """Delegate attribute access to FB2Parser instance."""
        return getattr(self._parser, name)

    # ================== PUBLIC ==================

    @staticmethod
    def list_speakers() -> List[str]:
        """List available speakers from Silero TTS models."""
        speakers = []
        for model_speakers in Book2Voice.SILERO_MODELS.values():
            speakers.extend(model_speakers)
        return speakers

    @staticmethod
    def get_speaker_model(speaker: str) -> Optional[str]:
        """Get model name for a given speaker."""
        for model_name, speakers in Book2Voice.SILERO_MODELS.items():
            if speaker in speakers:
                return model_name
        return None

    def gen(self, speaker: str = "eugene", output_dir: str = "") -> None:
        """Generate audiobook from FB2 file."""
        model_name = self.get_speaker_model(speaker)
        if not model_name:
            speakers = "\n" + "\n".join(self.list_speakers())
            raise ValueError(f"Speaker '{speaker}' not found in: {speakers}")
        self._load_tts_model(model_name)

        Book2Voice._resampler = torchaudio.transforms.Resample(
            self.SAMPLE_TTS_RATE, self.SAMPLE_AUDIO_RATE
        )

        # Prepare output directory
        authors = ", ".join(self.get_authors()) or "Unknown Author"
        book_title = self.get_title() or "Unknown Title"
        if not output_dir:
            output_dir = os.path.join(os.path.dirname(self.path), "audiobooks")
        out_root = os.path.join(output_dir, authors, book_title)
        os.makedirs(out_root, exist_ok=True)

        # Prepare for parallel processing
        workers = max(1, cpu_count() // 2)
        volumes = self._flatten_parts()
        total_chapters = sum(len(v) for v in volumes.values())
        total_chunks = sum(
            len(pp)
            for v in volumes.values()
            for p in v.values()
            for pp in p
        )
        logger.info(
            f'Creating audiobook with {workers} workers: '
            f'{total_chapters} chapters, {total_chunks} chunks'
        )

        # Setup multiprocessing and progress bar
        manager = Manager()
        progress_q = manager.Queue()
        pbar = tqdm(
            total=total_chunks, desc="Progress", unit="chunk", ncols=80
        )
        progress_batch = min(max(1, total_chunks // 200), 100)

        # Prepare task list for Pool
        tasks = []
        idx_volume = 1
        idx_chapter = 1
        for volume, chapters in volumes.items():
            if volume is None:
                base = out_root
            else:
                base = os.path.join(out_root, f"{idx_volume:02d} - {volume}")
            os.makedirs(base, exist_ok=True)
            idx_volume += 1
            for title, paragraphs in chapters.items():
                tasks.append(
                    (
                        idx_chapter,
                        progress_q,
                        progress_batch,
                        authors,
                        book_title,
                        self._book.get("cover"),
                        title,
                        paragraphs,
                        base,
                        speaker,
                        model_name,
                    )
                )
                idx_chapter += 1

        # Start progress listener thread
        listener = threading.Thread(
            target=self._progress_listener,
            args=(progress_q, pbar),
            daemon=True
        )
        listener.start()

        # Process chapters in parallel
        try:
            with Pool(workers) as pool:
                pool.map(Book2Voice._generate_chapter_audio, tasks)
        except Exception as e:
            logger.error(f"Error during audio generation: {e}")
            raise
        finally:
            progress_q.put(None)
            listener.join()
            pbar.close()

        logger.info(f"Audiobook created successfully in '{out_root}'")

    # ================== PRIVATE ==================

    def _flatten_parts(
        self,
    ) -> Dict[Optional[str], Dict[str, List[List[str]]]]:
        """Convert hierarchical book structure to flat chapter list."""
        result: Dict[Optional[str], Dict[str, List[List[str]]]] = {}

        # Recursive function to walk through parts and chapters
        def walk(part, prefix="") -> Tuple[List[List[str]], str]:
            title = part.get("title") or prefix or "Chapter"
            full_title = title if not prefix else f"{prefix}. {title}"

            paragraphs = []
            for p in part.get("paragraphs", []):
                chunks = self._split_to_chunks(p)
                if chunks:
                    paragraphs.append(chunks)
            return paragraphs, full_title

        # Process each part
        for part in self._book.get("parts", []):
            if part.get("chapters"):
                volume = part.get("title")
                result[volume] = {}
                for ch in part["chapters"]:
                    paragraphs, title = walk(ch)
                    if paragraphs:
                        result[volume][title] = paragraphs
            else:
                result.setdefault(None, {})
                paragraphs, title = walk(part)
                if paragraphs:
                    result[None][title] = paragraphs

        return result

    def _split_to_chunks(self, text: str) -> List[str]:
        """Split text into chunks for synthesis (max MAX_LEN chars each)."""
        sentences = self.SENTENCE_RE.split(text)
        chunks, buf = [], ""

        for s in sentences:
            if len(buf) + len(s) + 1 <= self.MAX_LEN:
                buf = f"{buf} {s}".strip()
            else:
                chunks.append(buf)
                buf = s

        if buf:
            chunks.append(buf)

        return chunks

    def _load_tts_model(self, name: str) -> None:
        """Load TTS model if not already loaded."""
        model_filepath = os.path.join(self.MODEL_PATH, name)
        model_url = self.SILERO_MODEL_URL + "/" + name
        if os.path.isfile(model_filepath):
            return
        logger.info(f"Downloading TTS model from {model_url}...")
        try:
            os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
            torch.hub.download_url_to_file(model_url, model_filepath)
            logger.info("Model downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    @staticmethod
    def _get_tts_model(name: str) -> torch.nn.Module:
        """Load TTS model for static method use."""
        model = torch.package.PackageImporter(
            os.path.join(Book2Voice.MODEL_PATH, name)
        ).load_pickle("tts_models", "model")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        return model

    @staticmethod
    def _write_id3(mp3_path: str, title: str, author: str, book_title: str,
                   cover: Optional[Dict]) -> None:
        """Write ID3 metadata to MP3 file."""
        logger.debug(f"Writing ID3 tags to '{mp3_path}': title='{title}', "
                     f"author='{author}', book_title='{book_title}'")
        tags = ID3()
        tags.add(TIT2(encoding=3, text=title))
        tags.add(TPE1(encoding=3, text=author))
        tags.add(TALB(encoding=3, text=book_title))

        if cover and "data" in cover:
            tags.add(
                APIC(
                    encoding=3,
                    mime=f"image/{cover.get('ext', 'jpeg')}",
                    type=3,
                    desc="Cover",
                    data=cover.get("data"),
                )
            )

        tags.save(mp3_path)

    @staticmethod
    def _progress_listener(queue, pbar) -> None:
        """Listen queue for progress updates and update progress bar."""
        while True:
            msg = queue.get()
            if msg is None:
                break
            pbar.update(msg)

    @staticmethod
    def _normalize_audio(audio):
        """
        Normalize audio to prevent clipping and ensure consistent volume.
        """
        audio = Book2Voice._resampler(audio)
        audio = torch.clamp(audio, -0.98, 0.98)

        return audio.cpu().numpy()

    @staticmethod
    def _generate_chapter_audio(args: Tuple) -> str:
        """Generate audio for a single chapter.
        Used for parallel processing in Pool.
        """
        (
            idx,
            progress_q,
            progress_batch,
            author,
            book_title,
            cover,
            chapter,
            paragraphs,
            out_dir,
            speaker,
            model_name,
        ) = args

        logger.info(f"Generating audio for chapter '{chapter}'")
        model = Book2Voice._get_tts_model(model_name)
        normalizer = Normalizer()

        # Prepare output WAV path
        chapter_save = re.sub(
            r'[<>:"/\\|?*]',
            '_',
            f"{idx:02d} - {chapter}"
        )
        chapter_path = os.path.join(out_dir, chapter_save + ".mp3")

        try:
            with sf.SoundFile(
                chapter_path,
                mode='w',
                samplerate=Book2Voice.SAMPLE_AUDIO_RATE,
                channels=1,
                subtype="MPEG_LAYER_III",
                format='MP3',
                compression_level=0.6,
            ) as out_f:
                title = (
                    f"Вы слушаете аудиокнигу '{book_title}'... "
                    f"Автор {author}... {chapter}"
                    if idx == 1
                    else chapter
                )
                with torch.inference_mode():
                    audio = model.apply_tts(
                        text=normalizer.normalize_text(title),
                        speaker=speaker,
                        sample_rate=Book2Voice.SAMPLE_TTS_RATE,
                    )
                audio = Book2Voice._normalize_audio(audio)
                out_f.write(audio)
                pause_samples = int(
                    Book2Voice.PAUSE_PARAGRAPH * Book2Voice.SAMPLE_AUDIO_RATE
                )
                out_f.write(
                    torch.zeros(pause_samples)
                )

                batch = 0
                for paragraph in paragraphs:
                    pause = (
                        Book2Voice.PAUSE_DIALOG
                        if (paragraph and
                            paragraph[0].startswith(("—", "–", "-")))
                        else Book2Voice.PAUSE_PARAGRAPH
                    )
                    out_f.write(
                        torch.zeros(int(pause * Book2Voice.SAMPLE_AUDIO_RATE))
                    )

                    for chunk in paragraph:
                        with torch.inference_mode():
                            audio = model.apply_tts(
                                text=normalizer.normalize_text(chunk),
                                speaker=speaker,
                                sample_rate=Book2Voice.SAMPLE_TTS_RATE,
                            )
                        audio = Book2Voice._normalize_audio(audio)
                        out_f.write(audio)

                        batch += 1
                        if batch >= progress_batch:
                            progress_q.put(batch)
                            batch = 0

                if batch:
                    progress_q.put(batch)

            Book2Voice._write_id3(
                chapter_path, chapter_save, author, book_title, cover
            )

        except Exception as e:
            logger.error(f"Error generating audio for chapter {idx}: {e}")
            raise

        return chapter_path
