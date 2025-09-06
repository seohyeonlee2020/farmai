#pdf to training data pipeline
import os
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path
import re
import unicodedata

# Assign directory
directory = "./farmai_training_data"

def extract_text(directory):
	text_data = {}
	for name in os.listdir(directory):
		print(f'processing {name}')
		filename = os.path.join(directory, name)
		try:
			reader = PdfReader(filename)
			page_list = reader.pages
			text_content = "".join(page.extract_text() for page in page_list)
		except:
			images = convert_from_path(filename, 100)
			print(f"Converted {len(images)} pages in {name}")

			for i, img in enumerate(images):
			    # OCR each page image
				text = pytesseract.image_to_string(img)
			text_content += text
		text_data[name] = text_content
	return text_data

def clean_text(text):
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Split joined words
    text = re.sub(r'-\s+', '', text)  # Fix hyphenated line breaks

    # Remove page numbers, headers, footers (customize patterns)
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)

    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)

    return text

def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def extract_metadata(pdf_path, chunk_text, chunk_index):
    return {
        'source_file': pdf_path,
        'chunk_id': chunk_index,
        'chunk_length': len(chunk_text),
        'word_count': len(chunk_text.split()),
    }

def is_repetitive(text):
    words = text.split()
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < 0.3  # Adjust threshold

def filter_chunks(chunks):
    filtered = []
    for chunk in chunks:
        # Remove chunks that are too short or too repetitive
        if len(chunk.split()) < 20:
            continue
        if is_repetitive(chunk):
            continue
        filtered.append(chunk)
    return filtered


#text_data =  extract_text(directory)






