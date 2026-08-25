import pdfplumber
import io


def read_uploaded_bytes(filename, content_bytes):
    name = filename.lower()

    if name.endswith(".pdf"):
        text = []
        with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text).strip()

    elif name.endswith(".txt"):
        return content_bytes.decode("utf-8").strip()

    else:
        return ""
