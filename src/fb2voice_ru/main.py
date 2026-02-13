import argparse
import logging
import sys
from pathlib import Path

from .logger import setup_logging
from .tts import Book2Voice


def main() -> int:
    parser = argparse.ArgumentParser(
        prog='fb2voice-ru',
        description='Convert FB2 e-books to MP3 audiobooks',
        epilog='Examples:\n'
               '  fb2voice-ru book.fb2\n'
               '  fb2voice-ru book.fb2 --speaker baya --output audiobooks\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'book',
        help='Path to FB2 file'
    )
    parser.add_argument(
        '--speaker',
        default='eugene',
        choices=['aidar', 'baya', 'kseniya', 'xenia', 'eugene'],
        help='Speaker for speech synthesis (default: eugene)'
    )
    parser.add_argument(
        '--output',
        help='Output directory for generated audiobook'
             '(default: <input_dir>/audiobooks)'
    )
    parser.add_argument(
        '--log-level',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='INFO',
        help='Logging level'
    )
    args = parser.parse_args()

    # Update logging level
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Check if file exists and is FB2 format
    book_path = Path(args.book)
    if not book_path.exists():
        logger.error(f"File not found: {args.book}")
        return 1
    if not book_path.suffix.lower() == '.fb2':
        logger.error(
            f"File '{args.book}' not be FB2 format "
            f"(extension: {book_path.suffix})"
        )
        return 1

    try:
        logger.info(f"Processing file: {args.book}")
        converter = Book2Voice(str(book_path))
        converter.gen(speaker=args.speaker, output_dir=args.output)
        return 0
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
