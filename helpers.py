# Standard Libraries
import re
import json

# Third Party Libraries
from IPython.display import Markdown
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WikipediaLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CACHE_FILE = "wikipedia_cache.json"
store = {}

def to_markdown(text):
    """
    Convert text to Markdown format:
    - Replace bullet points (•) with Markdown lists (* item)
    - Preserve Python code blocks correctly
    - Add blockquote formatting to non-code text

    Args:
        text (str): Input text to format

    Returns:
        Markdown: Formatted markdown-compatible text
    """
    # Replace bullet points (•) with Markdown-compatible lists (* item)
    text = text.replace('•', '  * ')

    # Function to preserve code blocks
    def preserve_code(match):
        return f"\n```python\n{match.group(1)}\n```\n"

    # Extract and preserve Python code blocks
    text = re.sub(r"```python\n(.*?)\n```", preserve_code, text, flags=re.DOTALL)

    # Split text into lines for better processing
    lines = text.split("\n")
    formatted_lines = []
    inside_code_block = False

    for line in lines:
        # Detect start and end of a code block
        if line.startswith("```"):
            inside_code_block = not inside_code_block
            formatted_lines.append(line)
            continue

        # Apply blockquote formatting **only** to non-code lines
        if not inside_code_block:
            line = line.strip() # Remove leading/trailing whitespace from each line
            if line:  # Only add non-empty lines
                formatted_lines.append(f"> {line}")
        else:
            formatted_lines.append(line)

    # Join lines back into a full formatted text
    formatted_text = "\n".join(formatted_lines)
    
    return Markdown(formatted_text)

def format_docs(docs):
    """Formats a list of documents into a single string.

    Combines the page content of each document in the input list
    into a single string, separated by double newline characters.

    Args:
        docs: A list of Document objects, where each object is 
              expected to have a 'page_content' attribute.

    Returns:
        A single string containing the concatenated page content
        of all documents, separated by double newlines.  Returns an
        empty string if the input list is empty or None.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates a chat message history for a given session ID.

    Args:
        session_id: The ID of the chat session.

    Returns:
        A BaseChatMessageHistory object for the session.
    """
    display(to_markdown(f"#### Store: \n\n {store}"))
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def load_wikipedia_with_cache(title):
    """Loads a Wikipedia page, using a local cache to speed up subsequent requests.

    First checks if the Wikipedia page for the given title is already
    present in the local cache file (wikipedia_cache.json). If it is, the cached
    documents are returned directly. Otherwise, it loads the page from Wikipedia,
    stores the documents in the cache, and then returns them.

    Args:
        title: The title of the Wikipedia page to load.

    Returns:
        A list of Document objects representing the Wikipedia page content.
    """
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}

    if title in cache:
        # Convert cached dictionaries back to Document objects
        return [Document(**doc_dict) for doc_dict in cache[title]]

    loader = WikipediaLoader(title)
    docs = loader.load()

    # Convert Document objects to dictionaries for JSON serialization
    cache[title] = [doc.__dict__ for doc in docs]  # Use __dict__ for simple conversion

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

    return docs

def split_text_documents(docs, chunk_size=1000, chunk_overlap=200):
    """
    Splits a list of Document objects into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        docs (list of Document): A list of LangChain Document objects to be split.
        chunk_size (int): Maximum size of each chunk in characters.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        list of Document: A list of smaller Document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return text_splitter.split_documents(docs)

def display_answer(question: str, answer: str):
    """
    Displays the question and answer in a formatted Markdown style.

    Args:
        question (str): The question being asked.
        answer (str): The answer retrieved from the model or RAG system.

    Output:
        Displays the question as a Markdown header and the answer below it 
        in a nicely formatted manner within a Jupyter Notebook.
    """
    display(to_markdown(f"#### {question}"))
    display(to_markdown(answer))


