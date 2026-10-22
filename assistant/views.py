
import os
import re
import numpy as np

from django.shortcuts import render
from pypdf import PdfReader
import numpy as np


# Demo storage: suitable for one local hackathon session
DOCUMENT_CHUNKS = []
DOCUMENT_NAME = ""
EMBEDDINGS = None
MODEL = None


def split_into_chunks(text, size=900, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []

    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def read_document(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[Page {page_number}] {page_text}")

        return "\n".join(pages)

    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")

    raise ValueError("Please upload a PDF or TXT file.")


def retrieve_passages(question, top_k=3):
    if not DOCUMENT_CHUNKS:
        return []

    question_words = set(re.findall(r"\w+", question.lower()))
    results = []

    for index, chunk in enumerate(DOCUMENT_CHUNKS):
        chunk_words = set(re.findall(r"\w+", chunk.lower()))

        if not question_words:
            continue

        score = len(question_words & chunk_words) / len(question_words)

        if score >= 0.20:
            results.append({
                "text": chunk,
                "score": round(score, 3),
                "source": f"Passage {index + 1}",
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]




def generate_answer(question, passages):
    if not passages:
        return (
            "I couldn't find enough relevant information "
            "in the document to answer this question."
        )

    import re

    question_words = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
    )

    stop_words = {
        "what", "which", "when", "where", "who", "whom",
        "why", "how", "does", "did", "was", "were",
        "are", "the", "and", "for", "from", "that",
        "this", "with", "their", "they", "them", "has",
        "have", "had", "can", "could", "would", "should",
        "explain", "describe", "write", "short"
    }

    question_words -= stop_words

    candidates = []

    for passage in passages:
        sentences = re.split(
            r'(?<=[.!?])\s+',
            passage["text"].replace("\n", " ")
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if len(sentence.split()) < 6:
                continue

            sentence_words = set(
                re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
            )

            matches = question_words & sentence_words

            if question_words:
                score = len(matches) / len(question_words)
            else:
                score = 0

            if score > 0:
                candidates.append({
                    "text": sentence,
                    "score": score,
                    "source": passage["source"]
                })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    selected = []
    seen = set()

    for item in candidates:
        normalized = item["text"].lower()

        if normalized in seen:
            continue

        seen.add(normalized)
        selected.append(item)

        if len(selected) == 3:
            break

    if not selected:
        return (
            "I couldn't identify a sufficiently relevant sentence. "
            "Try asking the question using keywords from the document."
        )

    answer = "Based on the document:\n\n"

    for item in selected:
        answer += f"{item['text']} [{item['source']}]\n\n"

    return answer.strip()


    return answer


    prompt = f"""
You are DocuMind, a document question-answering assistant.

Answer the question using ONLY the supplied document passages.
If the passages do not contain enough information, say so.
Do not invent facts or citations.
Refer to the supporting passage labels in your answer.

DOCUMENT PASSAGES:
{context}

USER QUESTION:
{question}
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text or "The AI returned an empty response."

    except Exception as error:
        print("Gemini error type:", type(error).__name__)
        print("Gemini error details:", str(error)[:1500])

    return (
        "AI generation failed. Check the VS Code terminal "
        "for the detailed error. Retrieved passages are shown below."
    )


def home(request):
    global DOCUMENT_CHUNKS, DOCUMENT_NAME, EMBEDDINGS

    answer = ""
    passages = []
    error = ""
    question = ""

    if request.method == "POST":
        print("DEBUG: POST request received")
        question = request.POST.get("question", "").strip()
        uploaded_file = request.FILES.get("document")

        try:
            if uploaded_file:
                if uploaded_file.size > 10 * 1024 * 1024:
                    raise ValueError("Please use a file smaller than 10 MB.")

                document_text = read_document(uploaded_file)

                if not document_text.strip():
                    raise ValueError(
                        "No extractable text found. Scanned PDFs "
                        "may require OCR, which this demo does not include."
                    )

                DOCUMENT_CHUNKS = split_into_chunks(document_text)
                DOCUMENT_NAME = uploaded_file.name

                EMBEDDINGS = None

            if question:
                if not DOCUMENT_CHUNKS:
                    raise ValueError(
                        "Upload a document before asking a question."
                    )

                passages = retrieve_passages(question)

                if not passages:
                    answer = (
                        "I could not find sufficiently relevant passages "
                        "in the uploaded document. Try asking a different "
                        "question or uploading a more relevant document."
                    )
                else:
                    answer = generate_answer(question, passages)

            elif not uploaded_file:
                raise ValueError("Upload a PDF or TXT file first.")

        except Exception as exc:
            error = str(exc)

    return render(request, "assistant/index.html", {
        "answer": answer,
        "passages": passages,
        "error": error,
        "question": question,
        "document_name": DOCUMENT_NAME,
    })
