"""Simple example of using Hapax with an LLM for text processing."""
from typing import List
import os
from dotenv import load_dotenv
from hapax import ops, Graph, set_openlit_config
import openai
import openlit

# Load environment variables
load_dotenv()

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Configure OpenLit globally
try:
    openlit.init(
        otlp_endpoint=os.getenv("OPENLIT_ENDPOINT", "http://localhost:4318"),
        environment=os.getenv("OPENLIT_ENVIRONMENT", "development"),
        application_name=os.getenv("OPENLIT_APP_NAME", "hapax_llm_example"),
        trace_content=True
    )
    set_openlit_config({
        "trace_content": True,
        "disable_metrics": False
    })
except Exception as e:
    print(f"Warning: OpenLit initialization failed: {e}")
    print("Continuing without OpenLit monitoring...")
    set_openlit_config(None)

@ops(
    name="clean_text",
    tags=["preprocessing"]
)
def clean_text(text: str) -> str:
    """Clean and normalize input text."""
    return text.strip().lower()

@ops(
    name="split_into_chunks",
    tags=["preprocessing"]
)
def split_into_chunks(text: str) -> List[str]:
    """Split text into chunks of reasonable size."""
    # Simple splitting by sentences for this example
    chunks = [s.strip() for s in text.split('.') if s.strip()]
    print(f"Split into {len(chunks)} chunks: {chunks}")
    return chunks

@ops(
    name="summarize_chunk",
    tags=["llm"]
)
def summarize_chunk(chunk: str) -> str:
    """Summarize a chunk of text using OpenAI API."""
    print(f"Summarizing chunk: {chunk}")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
            {"role": "user", "content": f"Please summarize this text concisely: {chunk}"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    summary = response.choices[0].message.content.strip()
    print(f"Summary: {summary}")
    return summary

@ops(
    name="combine_summaries",
    tags=["postprocessing"]
)
def combine_summaries(summaries: List[str]) -> str:
    """Combine individual summaries into a final summary."""
    print(f"Combined summaries: {' '.join(summaries)}")
    return " ".join(summaries)

def create_summary_pipeline() -> Graph[str, str]:
    """Create the text summarization pipeline using the fluent API."""
    return (
        Graph("text_summarization", "Process and summarize text using an LLM")
        .then(clean_text)
        .then(split_into_chunks)
        .then(summarize_chunk)  # Auto-maps over list of chunks
        .then(combine_summaries)
    )

def summarize_text(text: str) -> str:
    """Process and summarize text using an LLM."""
    pipeline = create_summary_pipeline()
    summary = pipeline.execute(text)
    print(f"\nFinal summary: {summary}")
    return summary

# Example usage
if __name__ == "__main__":
    text = """
    The Python programming language was created by Guido van Rossum and released in 1991. 
    It emphasizes code readability with its notable use of significant whitespace. 
    Python features a dynamic type system and automatic memory management.
    """
    
    summary = summarize_text(text)
    print("\nOriginal text:")
    print(text)
    print("\nSummary:")
    print(summary)
