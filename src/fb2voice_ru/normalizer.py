import logging
import re

from num2words import num2words
from silero_stress import load_accentor

# Setup logging
logger = logging.getLogger(__name__)


class Normalizer:
    """Class for normalizing text and audio for TTS synthesis."""
    # Regular expressions for text normalization
    SPACE_RE = re.compile(r"\s+")
    NUMBER_RE = re.compile(r"(?<!\w)(-?(?:\d+\.\d+|\d+))")
    PERCENT_RE = re.compile(r"(?<!\w)(-?(?:\d+\.\d+|\d+))\s*%")
    NUMERO_RE = re.compile(r"№\s*(\d+)")
    ROMAN_RE = re.compile(
        r"\b(?=[MDCLXVI]+\b)"
        r"(CM|CD|D?C{0,3})"
        r"(XC|XL|L?X{0,3})"
        r"(IX|IV|V?I{0,3})\b",
        re.IGNORECASE
    )
    LATIN_TEXT_RE = re.compile(r'\b[A-Za-z]+\b')

    # Abbreviations to expand
    ABBREVIATIONS = [
        (re.compile(r"\bт\.к\.", re.IGNORECASE), "так как"),
        (re.compile(r"\bт\.е\.", re.IGNORECASE), "то есть"),
        (re.compile(r"\bт\.д\.", re.IGNORECASE), "так далее"),
        (re.compile(r"\bт\.п\.", re.IGNORECASE), "тому подобное"),
        (re.compile(r"\bдр\.", re.IGNORECASE), "другие"),
        (re.compile(r"\bгг\.", re.IGNORECASE), "годы"),
        (re.compile(r"\bн\.э\.", re.IGNORECASE), "нашей эры"),
        (re.compile(r"\bкв\.", re.IGNORECASE), "квартира"),
        (re.compile(r"\bд\.", re.IGNORECASE), "дом"),
        (re.compile(r"\bул\.", re.IGNORECASE), "улица"),
        (re.compile(r"\bпр\.", re.IGNORECASE), "проспект"),
        (re.compile(r"\bпл\.", re.IGNORECASE), "площадь"),
        (re.compile(r"\bг\.", re.IGNORECASE), "город"),
        (re.compile(r"\bим\.", re.IGNORECASE), "имени"),
        (re.compile(r"\bг-н\.", re.IGNORECASE), "господин"),
        (re.compile(r"\bг-жа\.", re.IGNORECASE), "госпожа"),
    ]

    def __init__(self):
        """Initialize the normalizer."""
        self.accentor = load_accentor()

    @staticmethod
    def roman_to_int(roman: str) -> str:
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
    def plural_percent(n: float) -> str:
        """Correctly pluralize 'percent' on the given number."""
        if isinstance(n, float) and not n.is_integer():
            return "процента"

        n = abs(int(n)) % 100
        n1 = n % 10
        if 10 < n < 20:
            return "процентов"
        if 1 < n1 < 5:
            return "процента"
        if n1 == 1:
            return "процент"

        return "процентов"

    @staticmethod
    def phonetic_word(word: str) -> str:
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

    def normalize_text(self, text: str) -> str:
        """Normalize text before speech synthesis."""
        text = self.SPACE_RE.sub(" ", text).strip()
        text = text.replace("—", "-")
        text = self.ROMAN_RE.sub(
            lambda m: self.roman_to_int(m.group(0)), text
        )
        text = self.LATIN_TEXT_RE.sub(
            lambda m: self.phonetic_word(m.group(0)), text
        )
        text = self.NUMERO_RE.sub(r"номер \1", text)
        text = self.PERCENT_RE.sub(
            lambda m: (
                f"{num2words(m.group(1), lang='ru')} "
                f"{self.plural_percent(float(m.group(1)))}"
            ),
            text,
        )
        text = self.NUMBER_RE.sub(
            lambda m: num2words(m.group(1), lang="ru"),
            text,
        )

        for pattern, replacement in self.ABBREVIATIONS:
            text = pattern.sub(replacement, text)

        return self.accentor(text)
