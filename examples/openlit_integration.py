"""
Example demonstrating the integration of Hapax with OpenLIT for GPU monitoring and evaluations.

This example shows how to:
1. Enable GPU monitoring for a Hapax graph
2. Use OpenLIT evaluations with Hapax operations
3. Add evaluation to a graph execution

Requirements:
- openlit>=1.33.8
- openai>=1.0.0 (if using OpenAI for evaluations)
- anthropic>=0.5.0 (if using Anthropic for evaluations)
"""
import os
from typing import List
from dotenv import load_dotenv
from hapax import ops, Graph, eval, enable_gpu_monitoring

# Load .env file if present
load_dotenv()

# Global GPU monitoring (optional, can also be done per-graph)
enable_gpu_monitoring(
    otlp_endpoint=os.getenv("OPENLIT_ENDPOINT", "http://localhost:4318"),
    sample_rate_seconds=2,
    application_name="hapax_openlit_example",
    environment="development"
)

# Example 1: Using OpenLIT evaluations with the @eval decorator
@eval(
    evals=["hallucination"],
    threshold=0.7,
    use_openlit=True,  # This enables OpenLIT evaluations
    openlit_provider="openai",  # Can also use "anthropic"
    metadata={
        "contexts": ["Einstein won a Nobel Prize in Physics in 1921."],
        "prompt": "When did Einstein win the Nobel Prize?"
    }
)
def generate_einstein_fact(prompt: str) -> str:
    """Generate a fact about Einstein."""
    # This would normally use an LLM, but we'll hardcode for this example
    return "Einstein won the Nobel Prize in 1922."  # Deliberately slightly wrong date!

# Example 2: Create operations for a graph
@ops(name="clean_text", tags=["preprocessing"])
def clean_text(text: str) -> str:
    """Clean and normalize input text."""
    return text.strip().lower()

@ops(name="extract_entities", tags=["nlp"])
def extract_entities(text: str) -> List[str]:
    """Extract main entities from text."""
    # Simplified entity extraction
    words = text.split()
    entities = [word for word in words if word[0].isupper() or word in ["i", "me", "you"]]
    return entities if entities else ["<no entities found>"]

@ops(name="generate_response", tags=["llm"])
def generate_response(query: str) -> str:
    """Generate a response using an LLM."""
    # This would normally call an actual LLM, but we'll simulate for this example
    if "einstein" in query.lower():
        return "Albert Einstein received the Nobel Prize in Physics in 1921 for his discovery of the photoelectric effect."
    elif "award" in query.lower() or "prize" in query.lower():
        return "The Nobel Prize is awarded for achievements in Physics, Chemistry, Medicine, Literature, Peace, and Economics."
    else:
        return "I don't have specific information about that query."

def main():
    print("Hapax + OpenLIT Integration Example\n")
    
    # Example 1: Using the @eval decorator with OpenLIT
    print("Example 1: Using @eval decorator with OpenLIT evaluations")
    try:
        result = generate_einstein_fact("Tell me about Einstein's Nobel Prize")
        print(f"Result: {result}")
        print("Evaluation passed!")
    except Exception as e:
        print(f"Evaluation failed: {e}")
    
    print("\n" + "-" * 60 + "\n")
    
    # Example 2: Using with_gpu_monitoring and with_evaluation in a graph
    print("Example 2: GPU Monitoring and Evaluation in a Graph")
    
    # Create a query pipeline with GPU monitoring and evaluation
    pipeline = (
        Graph("query_pipeline", "Process and respond to user queries")
        .then(clean_text)
        .then(generate_response)
        .with_gpu_monitoring(enabled=True, sample_rate_seconds=1)
        .with_evaluation(
            eval_type="hallucination",
            threshold=0.6,
            provider="openai",
            fail_on_evaluation=False,  # Don't fail on evaluation to show results
            custom_config={
                "collect_metrics": True
            }
        )
    )
    
    # Execute the pipeline
    query = "When did Einstein receive the Nobel Prize?"
    result = pipeline.execute(query)
    
    print(f"Input: {query}")
    print(f"Output: {result}")
    
    # Display evaluation results if available
    if pipeline.last_evaluation:
        print("\nEvaluation Results:")
        for key, value in pipeline.last_evaluation.items():
            print(f"  {key}: {value}")
    
    # Get current GPU metrics
    try:
        from hapax import get_gpu_metrics
        gpu_metrics = get_gpu_metrics()
        if gpu_metrics:
            print("\nCurrent GPU Metrics:")
            for metric, value in gpu_metrics.items():
                print(f"  {metric}: {value}")
        else:
            print("\nNo GPU metrics available (either no GPU or monitoring not enabled).")
    except Exception as e:
        print(f"\nError getting GPU metrics: {e}")

if __name__ == "__main__":
    main() 