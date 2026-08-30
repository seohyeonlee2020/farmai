# Offline RAG chatbot that works on your laptop
Frontier AI models are only usable with high-speed internet and robust compute power. These requirements are not compatible with edge computing use cases where internet connectivity and cost are significant bottlenecks. Additionally, the majority of frontier models are hosted on clouds, which raise privacy concerns for use cases involving personal or proprietary information.

## How It Works
Everything stays offline.

**Setup (online, one-time, maybe intermittent upgrades)**:
- Extract text from 20-30 sources of your choice using pypdf (text OCR) and pytesseract (image extraction).
- Embed 20-30 PDFs using Hugging Face models.
- Create vectorstore.

**Offline use (laptops)**:
- small language model runs queries on embedded data. No internet needed.
- RAG setup: Model only uses retrieved docs to avoid hallucinations and false advice.
- All open-source, no vendor costs.

## Tech Choices for Offline Architecture
- **Small Language Model**: Lightweight enough for CPU on most laptops.
- **RAG only**: Sticks to input data to prevent hallucinations.
- **Open source components only**: Removes financial barriers and prevents vendor lock-in.
- English for MVP; will scale to multiple languages.

## Setup
```
# Install
git clone seohyeonlee2020/offline-rag-chatbot.git
pip install -r requirements.txt

# Run offline on localhost
streamlit run app.py
```

## Next Steps
- Usage documentation
- Multilingual support
- Group easily serchable information into a mass SMS service to reach users who do not have access to computers.



