"""Text-to-speech synthesis and audiobook generation.

Uses Silero TTS neural network to convert text to speech
and create MP3 audiobooks with metadata.
"""

import logging
import os
import re
import subprocess
import threading
import warnings
from multiprocessing import Manager, Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from mutagen.id3 import APIC, ID3, TALB, TIT2, TPE1
from num2words import num2words
from tqdm import tqdm

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
    MODEL_URL = 'https://models.silero.ai/models/tts/ru/v5_1_ru.pt'
    MODEL_PATH = torch.hub.get_dir() + '/.silero_ru_v5_1.pt'

    # Regular expressions for text normalization
    NUMBER_RE = re.compile(r"\d+")
    FLOAT_RE = re.compile(r"(\d+)[\.,](\d+)")
    NUMBER_SIGN_RE = re.compile(r"№\s*(\d+)")
    ROMAN_RE = re.compile(
        r"\b(?=[MDCLXVI]+\b)"
        r"(CM|CD|D?C{0,3})"
        r"(XC|XL|L?X{0,3})"
        r"(IX|IV|V?I{0,3})\b",
        re.IGNORECASE
    )
    PERCENT_RE = re.compile(r"(\d+)\s*%")
    SPACE_RE = re.compile(r"\s+")

    # Abbreviations to expand
    ABBREVIATIONS = [
        (re.compile(r"\bт\.к\.", re.IGNORECASE), "так как"),
        (re.compile(r"\bт\.е\.", re.IGNORECASE), "то есть"),
        (re.compile(r"\bт\.д\.", re.IGNORECASE), "так далее"),
        (re.compile(r"\bт\.п\.", re.IGNORECASE), "тому подобное"),
        (re.compile(r"\bдр\.", re.IGNORECASE), "другие"),
        (re.compile(r"\bгг\.", re.IGNORECASE), "годы"),
    ]

    SENTENCE_RE = re.compile(r'(?<=[.!?…])\s+')
    LATIN_TEXT_RE = re.compile(r'\b[A-Za-z]+\b')

    # Audio parameters
    SAMPLE_RATE = 48000
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
        self._tts_model = None

    def __getattr__(self, name):
        """Delegate attribute access to FB2Parser instance."""
        return getattr(self._parser, name)

    # ================== PUBLIC ==================

    def gen(self, speaker: str = "eugene", output_dir: str = "") -> None:
        """Generate audiobook from FB2 file."""
        self._load_tts_model()

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

    def _load_tts_model(self) -> None:
        """Load TTS model if not already loaded."""
        if self._tts_model is not None:
            return

        if not os.path.isfile(self.MODEL_PATH):
            logger.info(f"Downloading TTS model from {self.MODEL_URL}...")
            try:
                os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
                torch.hub.download_url_to_file(self.MODEL_URL, self.MODEL_PATH)
                logger.info("Model downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise

        logger.info("Loading TTS model...")
        try:
            self._tts_model = torch.package.PackageImporter(
                self.MODEL_PATH
            ).load_pickle("tts_models", "model")

            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            self._tts_model.to(device)
            logger.info(f"Model loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @staticmethod
    def _get_tts_model():
        """Load TTS model for static method use."""
        model = torch.package.PackageImporter(
            Book2Voice.MODEL_PATH
        ).load_pickle("tts_models", "model")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        return model

    @staticmethod
    def _wav_to_mp3(wav_path: str) -> str:
        """Convert WAV to MP3."""
        mp3_path = wav_path[:-4] + ".mp3"
        logger.debug(f"Converting WAV to MP3: '{wav_path}'")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", wav_path,
                    "-ac", "1",
                    "-ar", "22050",
                    "-b:a", "64k",
                    mp3_path,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.error(f"Failed to convert {wav_path} to MP3")
            raise
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg first.")
            raise

        os.remove(wav_path)
        return mp3_path

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
    def _phonetic_word(word: str) -> str:
        """Convert Latin word to Russian phonetic transcription."""
        combo_rules = sorted([
            ("sch", "ш"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"),
            ("ph", "ф"), ("th", "т"), ("qu", "кв"), ("ck", "к"),
            ("ng", "нг"), ("ts", "ц"), ("tz", "ц"), ("ya", "я"),
            ("yo", "ё"), ("yu", "ю"), ("ye", "е"),
        ], key=lambda x: -len(x[0]))
        letter_map = {
            "a": "а", "b": "б", "c": "к", "d": "д", "e": "е",
            "f": "ф", "g": "г", "h": "х", "i": "и", "j": "дж",
            "k": "к", "l": "л", "m": "м", "n": "н", "o": "о",
            "p": "п", "q": "к", "r": "р", "s": "с", "t": "т",
            "u": "у", "v": "в", "w": "в", "x": "кс", "y": "й",
            "z": "з",
        }
        word = word.lower()
        i = 0
        result = []

        while i < len(word):
            matched = False
            for pattern, repl in combo_rules:
                if word.startswith(pattern, i):
                    result.append(repl)
                    i += len(pattern)
                    matched = True
                    break
            if not matched:
                result.append(letter_map.get(word[i], word[i]))
                i += 1

        phonetic = ''.join(result)
        logger.debug(f"Converted Latin word '{word}' to phonetic '{phonetic}'")
        return phonetic

    @staticmethod
    def _roman_to_int(roman: str) -> str:
        """Convert Roman number to Arabic number."""
        roman_map = {
            "I": 1, "V": 5, "X": 10, "L": 50,
            "C": 100, "D": 500, "M": 1000,
        }
        total = 0
        prev = 0

        for ch in reversed(roman.upper()):
            value = roman_map[ch]
            if value < prev:
                total -= value
            else:
                total += value
                prev = value

        logger.debug(f"Converting Roman number '{roman}' to Arabic '{total}'")
        return str(total)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text before speech synthesis."""
        text = text.strip()
        text = text.replace("—", " — ")
        text = Book2Voice.ROMAN_RE.sub(
            lambda m: Book2Voice._roman_to_int(m.group(0)), text
        )
        text = Book2Voice.LATIN_TEXT_RE.sub(
            lambda m: Book2Voice._phonetic_word(m.group(0)), text
        )
        text = Book2Voice.NUMBER_SIGN_RE.sub(r"номер \1", text)
        text = Book2Voice.PERCENT_RE.sub(r"\1 процентов", text)

        for pattern, replacement in Book2Voice.ABBREVIATIONS:
            text = pattern.sub(replacement, text)

        text = Book2Voice.FLOAT_RE.sub(
            lambda m: f"{num2words(int(m.group(1)), lang='ru')} и "
                      f"{num2words(int(m.group(2)), lang='ru')}",
            text,
        )
        text = Book2Voice.NUMBER_RE.sub(
            lambda m: num2words(int(m.group()), lang="ru"),
            text,
        )

        text = Book2Voice.SPACE_RE.sub(" ", text).strip()

        return text

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
        ) = args

        logger.info(f"Generating audio for chapter '{chapter}'")
        model = Book2Voice._get_tts_model()

        # Prepare output WAV path
        chapter_save = re.sub(
            r'[<>:"/\\|?*]',
            '_',
            f"{idx:02d} - {chapter}"
        )
        chapter_path = os.path.join(out_dir, chapter_save + ".wav")

        try:
            with sf.SoundFile(
                chapter_path, 'w',
                Book2Voice.SAMPLE_RATE, 1,
                subtype='PCM_16'
            ) as out_f:

                # Initial chapter pause
                with torch.inference_mode():
                    audio = model.apply_tts(
                        text=Book2Voice._normalize_text(chapter),
                        speaker=speaker,
                        sample_rate=Book2Voice.SAMPLE_RATE,
                        put_accent=True,
                        put_yo=True,
                        put_stress_homo=True,
                        put_yo_homo=True,
                    )
                    out_f.write(audio)
                pause_samples = int(
                    Book2Voice.PAUSE_PARAGRAPH * Book2Voice.SAMPLE_RATE
                )
                out_f.write(
                    np.zeros(pause_samples, dtype='float32')
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
                        np.zeros(
                            int(pause * Book2Voice.SAMPLE_RATE),
                            dtype='float32'
                        )
                    )

                    for chunk in paragraph:
                        with torch.inference_mode():
                            audio = model.apply_tts(
                                text=Book2Voice._normalize_text(chunk),
                                speaker=speaker,
                                sample_rate=Book2Voice.SAMPLE_RATE,
                                put_accent=True,
                                put_yo=True,
                                put_stress_homo=True,
                                put_yo_homo=True,
                            )

                        out_f.write(audio)

                        batch += 1
                        if batch >= progress_batch:
                            progress_q.put(batch)
                            batch = 0

                if batch:
                    progress_q.put(batch)

            mp3_path = Book2Voice._wav_to_mp3(chapter_path)
            Book2Voice._write_id3(
                mp3_path, chapter_save, author, book_title, cover
            )

        except Exception as e:
            logger.error(f"Error generating audio for chapter {idx}: {e}")
            raise

        return mp3_path
