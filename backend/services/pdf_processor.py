"""
pdf_processor.py
----------------
Responsible for loading a PDF from disk and splitting it into
overlapping text chunks ready for embedding.

Uses:
  - PyPDFLoader   : extracts text page-by-page via pypdf
  - RecursiveCharacterTextSplitter : splits on paragraphs → sentences →
    words, falling back gracefully when the text is dense.

Returns a list of LangChain Document objects, each carrying:
  doc.page_content  : the chunk text
  doc.metadata      : { 'source': file_path, 'page': int }
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_and_split(file_path: str) -> list:
    """
    Load a PDF and return a flat list of Document chunks.

    Args:
        file_path: Absolute path to the saved PDF file on disk.

    Returns:
        List[Document] — each item is one text chunk with metadata.

    Raises:
        ValueError: if the PDF yields no extractable text (e.g. scanned image PDF).
        FileNotFoundError: if the path does not exist.
    """
    # Step 1: Load — PyPDFLoader reads each page and returns one Document per page.
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if not pages:
        raise ValueError(
            "PyPDFLoader returned no pages. "
            "The file may be empty or a scanned image PDF with no text layer."
        )

    # Step 2: Split — RecursiveCharacterTextSplitter tries to split on
    # ["\n\n", "\n", " ", ""] in order, preserving paragraph boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError(
            "Text splitting produced zero chunks. "
            "The PDF may contain only images or non-extractable content."
        )

    return chunks


def extract_texts(chunks: list) -> list[str]:
    """
    Pull the raw string content out of a list of Document objects.
    Used to pass plain strings to the embedder.

    Args:
        chunks: List[Document] from load_and_split()

    Returns:
        List[str] — the page_content of each chunk.
    """
    return [chunk.page_content for chunk in chunks]
