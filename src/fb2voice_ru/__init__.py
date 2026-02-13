__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .fb2 import FB2Parser
from .tts import Book2Voice

__all__ = ["FB2Parser", "Book2Voice"]
