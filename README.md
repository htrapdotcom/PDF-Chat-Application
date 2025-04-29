# PDF Chat Application

## Description

This Streamlit application enables users to interact with their PDF documents through a chat interface. It provides the functionality to:

- Upload PDF documents.
- Extract text from PDFs, including scanned documents using OCR.
- Ask questions about the PDF content.
- Receive answers based on the document's information.
- Manage multiple chat sessions.
- View and clear chat history.

The application utilizes the PyMuPDF, easyocr, and LangChain libraries, along with the Google Gemini API, to process documents and generate responses.

---

## Features

- **PDF Upload:** Users can upload PDF files for processing.
- **Text Extraction:** Extracts text from both digital and scanned PDFs.
- **OCR Support:** Uses easyocr to extract text from scanned documents.
- **Question Answering:** Allows users to ask questions about the uploaded PDF.
- **Multiple Chat Sessions:** Supports creating and switching between different chat sessions.
- **Chat History:** Maintains a history of questions and answers for each session.
- **Clear Chat History:** Provides an option to clear the chat history.

---

## Requirements

- Python 3.6 or higher
- pip (Python package installer)
- A Google API key

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Setup

### Clone the Repository

```bash
git clone [repository_url]
cd [repository_directory]
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Obtain a Google API Key

Get a Google Gemini API key from the [Google Cloud Console](https://console.cloud.google.com/).

### Set Up Environment Variables

Create a `.env` file in the project directory and add the following line:

```env
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

---

## Usage

Run the Streamlit application:

```bash
streamlit run work.py
```

### Workflow

1. Upload a PDF: Use the file uploader in the sidebar to upload your PDF document.
2. Ask Questions: Type your questions in the text input area and submit them.
3. View Answers: The application will display answers based on the PDF content.
4. Manage Chat Sessions: Use the sidebar to create new chat sessions or switch between existing ones.
5. View Chat History: Expand the chat history entries to see previous questions and answers.
6. Clear Chat History: Use the "Clear Chat History" button to remove the chat history for the current session.

---

## Code Explanation

### `work.py`

Main Streamlit app script handling:

- PDF uploads
- Text extraction
- OCR processing
- Question answering
- Chat session management

### `requirements.txt`

Specifies all necessary Python packages.

### `.env`

(Not included in repository) — Stores the Google API key securely as an environment variable.

---

## Future Enhancements

- Support for other document formats.
- Improved OCR accuracy.
- More advanced chat history management (e.g., saving/loading history).
- User authentication.

---

## Author

Parth

---

## License

[MIT License](https://opensource.org/licenses/MIT) or any other license you prefer.

