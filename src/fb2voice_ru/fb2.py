import base64
import logging
import os
from typing import Dict, List, Optional

from lxml import etree

# FB2 namespace
FB2_NS = {
    "fb": "http://www.gribuser.ru/xml/fictionbook/2.0",
    "xlink": "http://www.w3.org/1999/xlink",
}

logger = logging.getLogger(__name__)


class FB2Parser:
    """FB2 e-book parser.

    Extracts book metadata (title, authors, annotation, cover) and content
    (chapters, paragraphs, footnotes).
    """

    def __init__(self, path: str) -> None:
        """Initialize FB2 parser."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"FB2 file not found: {path}")
        self.root = etree.parse(path).getroot()
        self._notes: Optional[Dict[str, str]] = None

    def get_title(self) -> str:
        """Get book title."""
        el = self.root.find(".//fb:book-title", FB2_NS)
        if el is None or not el.text:
            return ""
        return el.text.strip().removesuffix(".")

    def get_authors(self) -> List[str]:
        """Get list of book authors."""
        authors = []
        for a in self.root.findall(".//fb:title-info/fb:author", FB2_NS):
            parts = []
            for tag in ("first-name", "middle-name", "last-name"):
                el = a.find(f"fb:{tag}", FB2_NS)
                if el is not None and el.text:
                    parts.append(el.text.strip())
            if parts:
                authors.append(" ".join(parts))
        return authors

    def get_annotation(self) -> str:
        """Get book annotation (description)."""
        paragraphs = self.root.findall(".//fb:annotation//fb:p", FB2_NS)
        return "\n".join(
            "".join(p.itertext()).strip()
            for p in paragraphs
            if (p.text or len(list(p)))
        )

    def get_book(self) -> Dict:
        """Get complete book structure with metadata and content.

        Returns a dictionary with keys: title, authors, annotation,
        cover, parts.
        """
        book = {
            "title": self.get_title(),
            "authors": self.get_authors(),
            "annotation": self.get_annotation(),
            "cover": self._collect_cover_info(),
            "parts": []
        }
        body = self.root.find("fb:body", FB2_NS)
        if body is None:
            return book

        for section in body.findall("fb:section", FB2_NS):
            part = self._parse_section(section)
            if part:
                book["parts"].append(part)
        return book

    def _collect_notes(self) -> Dict[str, str]:
        """Collect all footnotes from special notes section."""
        notes = {}
        notes_body = self.root.find(".//fb:body[@name='notes']", FB2_NS)
        if notes_body is None:
            return notes

        for section in notes_body.findall("fb:section", FB2_NS):
            note_id = section.attrib.get("id")
            if not note_id:
                continue
            paragraphs = section.findall(".//fb:p", FB2_NS)
            text = " ".join(
                "".join(p.itertext()).strip()
                for p in paragraphs
                if (p.text or len(list(p)))
            )
            notes[note_id] = text
        logger.debug(f"Collected {len(notes)} footnotes from notes section")
        return notes

    def _collect_cover_info(self) -> Optional[Dict]:
        """Extract cover image info from file."""
        cover_ref = self.root.find(".//fb:coverpage//fb:image", FB2_NS)
        if cover_ref is None:
            return None

        href = cover_ref.attrib.get(f"{{{FB2_NS['xlink']}}}href", "")
        image_id = href.lstrip("#")
        if not image_id:
            return None

        binary = self.root.find(f".//fb:binary[@id='{image_id}']", FB2_NS)
        if binary is None or not binary.text:
            return None

        content_type = binary.attrib.get("content-type", "image/jpeg")
        ext = content_type.split("/")[-1]
        data = base64.b64decode(binary.text)
        logger.debug(
            "Collected cover image: id=%s, type=%s, size=%d bytes",
            image_id,
            content_type,
            len(data),
        )
        return {"ext": ext, "data": data}

    def _parse_section(self, section) -> Optional[Dict]:
        """Recursively parse book section (chapter/volume)."""
        title_el = section.find("fb:title", FB2_NS)
        title = (
            ". ".join(title_el.itertext()).strip().removesuffix(".")
            if title_el is not None
            else None
        )
        logger.debug(f"Parsing section: title='{title}'")

        paragraphs = []
        for p in section.findall("fb:p", FB2_NS):
            paragraphs.append(self._extract_paragraph_with_notes(p))

        children = []
        for sub in section.findall("fb:section", FB2_NS):
            child = self._parse_section(sub)
            if child:
                children.append(child)

        if not title and not children and not paragraphs:
            return None

        return {
            "title": title,
            "paragraphs": paragraphs,
            "chapters": children
        }

    def _extract_paragraph_with_notes(self, p) -> str:
        """Extract paragraph text with footnotes substituted for links."""
        parts = []
        if p.text:
            parts.append(p.text)
        if self._notes is None:
            self._notes = self._collect_notes()
        for child in p:
            tag = etree.QName(child).localname
            if tag == "a":  # Link to footnote
                href = child.attrib.get(f"{{{FB2_NS['xlink']}}}href", "")
                note_id = href.lstrip("#")
                note_text = self._notes.get(note_id)
                if note_text:
                    logger.debug(f"Substituting footnote: id='{note_id}'")
                    parts.append(f" [примечание: {note_text}] ")
                if child.tail:
                    parts.append(child.tail)
            else:
                parts.append("".join(child.itertext()))
                if child.tail:
                    parts.append(child.tail)
        return "".join(parts).strip()
