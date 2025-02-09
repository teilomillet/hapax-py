"""Example usage of the Hapax graph system in a functional style."""
from typing import List, Dict
from hapax.core.decorators import ops

# Define individual operations
@ops(
    name="tokenize",
    description="Split text into words and normalize",
    tags=["preprocessing"]
)
def tokenize(text: str) -> List[str]:
    """Split text into words and normalize."""
    return text.lower().split()

@ops(
    name="normalize",
    description="Remove punctuation from words",
    tags=["preprocessing"]
)
def normalize(words: List[str]) -> List[str]:
    """Remove punctuation and normalize words."""
    return [word.strip('.,!?') for word in words]

@ops(
    name="remove_stops",
    description="Remove common stop words",
    tags=["preprocessing"]
)
def remove_stops(words: List[str]) -> List[str]:
    """Remove common stop words."""
    stops = {'the', 'a', 'an', 'and', 'or', 'but'}
    return [word for word in words if word.lower() not in stops]

@ops(
    name="count_frequencies",
    description="Count word frequencies",
    tags=["analysis"]
)
def count_frequencies(words: List[str]) -> Dict[str, int]:
    """Count frequency of each word."""
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

if __name__ == "__main__":
    # Create the text analysis pipeline
    pipeline = (
        tokenize 
        >> normalize 
        >> remove_stops 
        >> count_frequencies
    )
    
    # Generate visualization of the pipeline
    pipeline.visualize("text_analysis_pipeline")
    
    # Example usage
    text = "The quick brown fox jumps over the lazy dog. The fox is quick!"
    result = pipeline(text)
    print(f"Word frequencies: {result}")
