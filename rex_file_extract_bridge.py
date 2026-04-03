"""File extraction bridge for the chat UI.

Reads JSON from stdin:
  {"filename": "...", "data_base64": "...", "mime_type": "..."}

Writes JSON to stdout (single line):
  {"ok": true, "is_image": false, "extracted_text": "..."}
  {"ok": true, "is_image": true}
  {"ok": false, "error": "..."}
"""
from __future__ import annotations

import base64
import json
import sys

TEXT_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/csv",
    "text/x-markdown",
}

IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
}


def extract_pdf_text(data: bytes) -> str:
    """Attempt PDF text extraction using available libraries."""
    try:
        import fitz  # type: ignore[import]

        doc = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        pass

    try:
        import io

        from pdfminer.high_level import extract_text_to_fp  # type: ignore[import]
        from pdfminer.layout import LAParams  # type: ignore[import]

        output = io.StringIO()
        extract_text_to_fp(io.BytesIO(data), output, laparams=LAParams())
        return output.getvalue()
    except ImportError:
        pass

    return "(PDF text extraction unavailable — install PyMuPDF or pdfminer.six)"


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        filename = str(payload.get("filename", ""))
        data_base64 = str(payload.get("data_base64", ""))
        mime_type = str(payload.get("mime_type", "")).lower().split(";")[0].strip()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Bad input: {exc}"}), flush=True)
        sys.exit(1)

    try:
        data = base64.b64decode(data_base64)
    except Exception as exc:
        print(
            json.dumps({"ok": False, "error": f"Failed to decode base64: {exc}"}),
            flush=True,
        )
        sys.exit(1)

    if mime_type in IMAGE_MIME_TYPES:
        print(json.dumps({"ok": True, "is_image": True}), flush=True)
        return

    if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
        text = extract_pdf_text(data)
        print(
            json.dumps({"ok": True, "is_image": False, "extracted_text": text}),
            flush=True,
        )
        return

    # Text types and fallback
    if mime_type in TEXT_MIME_TYPES or filename.lower().endswith(
        (".txt", ".md", ".csv")
    ):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("latin-1")
            except Exception:
                text = "(Unable to decode file content)"
        print(
            json.dumps({"ok": True, "is_image": False, "extracted_text": text}),
            flush=True,
        )
        return

    # Unknown type — try as text
    try:
        text = data.decode("utf-8")
        print(
            json.dumps({"ok": True, "is_image": False, "extracted_text": text}),
            flush=True,
        )
    except Exception:
        print(
            json.dumps({"ok": False, "error": "Unsupported file type"}), flush=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
