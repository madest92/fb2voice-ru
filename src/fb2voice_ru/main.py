import argparse
import logging
import os
import sys
from pathlib import Path

from .logger import setup_logging
from .tts import Book2Voice


def parse_args():
    parser = argparse.ArgumentParser(
        prog='fb2voice-ru',
        description='Convert FB2 e-books to MP3 audiobooks',
        epilog='Examples:\n'
               '  fb2voice-ru --list-speakers\n'
               '  fb2voice-ru --sample --speaker baya --output audiobooks\n'
               '  fb2voice-ru book.fb2\n'
               '  fb2voice-ru book.fb2 --speaker baya --output audiobooks\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'book',
        nargs='?',
        help='Path to FB2 file'
    )
    parser.add_argument(
        '--list-speakers',
        action='store_true',
        help='List available speakers'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Generate a sample audio to preview the speaker voice'
    )
    parser.add_argument(
        '--speaker',
        default='eugene',
        help='Speaker for speech synthesis (default: eugene)'
    )
    parser.add_argument(
        '--output',
        help=(
            'Output directory for generated audiobook '
            '(default: <input_dir>/audiobooks)'
        )
    )
    parser.add_argument(
        '--log-level',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='INFO',
        help='Logging level'
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Update logging level
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # List speakers and exit
    if args.list_speakers:
        speakers = Book2Voice.list_speakers()
        print("Available speakers:")
        for speaker in speakers:
            print(f"  - {speaker}")
        return 0

    # Generate sample audiobooks and exit
    if args.sample:
        try:
            sample_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data/sample.fb2"
            )
            converter = Book2Voice(sample_path)
            speakers = ([args.speaker] if '--speaker' in sys.argv
                        else Book2Voice.list_speakers())
            logger.info(f"Generating sample audiobooks, speakers: {speakers}")
            for s in speakers:
                output_dir = (
                    os.path.join(args.output, s)
                    if args.output
                    else os.path.join(os.getcwd(), f"audiobooks/{s}")
                )
                converter.gen(speaker=s, output_dir=output_dir)
            return 0
        except ValueError as e:
            logger.error(f"Error: {e}")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return 1

    # Check if file exists and is FB2 format
    if not args.book:
        logger.error("Please provide a path to an FB2 file.")
        args.parser.print_help()
        return 1
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

    # Process convert the book to voice
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
