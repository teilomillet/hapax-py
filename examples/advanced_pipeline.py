"""Example of using Hapax for LLM-based text analysis.

This example shows how to:
1. Create reusable operations with @ops
2. Build a parallel processing graph
3. Handle LLM responses safely
4. Use type hints for safety
"""

import json
from typing import Dict, List, Union, Any
from openai import OpenAI
from dotenv import load_dotenv
from hapax import Graph, ops
from hapax.core.flow import Merge

# Load environment variables
load_dotenv()
client = OpenAI()

# Type aliases for clarity
Sentiment = Dict[str, Union[float, Dict[str, float]]]
Entity = Dict[str, str]

@ops(name="sentiment_analysis")
def analyze_sentiment(text: str) -> Sentiment:
    """Analyze text sentiment using LLM."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Return sentiment scores as JSON: {positive, negative, neutral, emotions:{joy, sadness, anger, surprise}}"
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    try:
        data = json.loads(response.choices[0].message.content)
        return {
            "positive": float(data.get("positive", 0.0)),
            "negative": float(data.get("negative", 0.0)),
            "neutral": float(data.get("neutral", 1.0)),
            "emotions": {
                "joy": float(data.get("emotions", {}).get("joy", 0.0)),
                "sadness": float(data.get("emotions", {}).get("sadness", 0.0)),
                "anger": float(data.get("emotions", {}).get("anger", 0.0)),
                "surprise": float(data.get("emotions", {}).get("surprise", 0.0))
            }
        }
    except Exception:
        return {
            "positive": 0.0, "negative": 0.0, "neutral": 1.0,
            "emotions": {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "surprise": 0.0}
        }

@ops(name="entity_extraction")
def extract_entities(text: str) -> List[Entity]:
    """Extract named entities using LLM."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Extract entities as: Entity Name (type). Types: person, organization, location, date, event"
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0
    )
    
    entities = []
    for line in response.choices[0].message.content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        try:
            # Try to extract entity name and type using parentheses
            name_part = line[:line.rindex('(')].strip()
            type_part = line[line.rindex('(')+1:line.rindex(')')].strip()
            
            if name_part and type_part:
                entities.append({
                    "name": name_part,
                    "type": type_part
                })
        except (ValueError, IndexError):
            # Skip malformed lines
            continue
            
    return entities

@ops(name="summarize")
def summarize(text: str) -> str:
    """Generate a concise summary using LLM."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Create a 1-2 sentence summary capturing the main points."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def combine_results(results: List[Any]) -> Dict[str, Any]:
    """Combine results from different branches."""
    return {
        "summary": results[0],
        "sentiment": results[1],
        "entities": results[2]
    }

# Build the analysis pipeline using Graph API
pipeline = (
    Graph("text_analysis")
    .branch(
        summarize,  # Branch 1: Generate summary
        analyze_sentiment,  # Branch 2: Analyze sentiment
        extract_entities  # Branch 3: Extract entities
    )
    .then(Merge("combine_results", combine_results))
)

# Visualize the pipeline
pipeline.visualize()

# Example usage
if __name__ == "__main__":
    texts = [
        """The SpaceX Starship, developed by Elon Musk's team, represents a major leap in space technology. 
        The massive rocket, designed for Mars missions, successfully completed its first integrated flight test 
        despite some challenges. NASA sees this as a crucial development for future lunar missions.""",
        
        """La Tour Eiffel, symbole emblématique de Paris, attire des millions de visiteurs chaque année. 
        Construite par Gustave Eiffel pour l'Exposition universelle de 1889, elle devait être temporaire 
        mais est devenue l'un des monuments les plus célèbres du monde."""
    ]
    
    print("\nAnalyzing texts:")
    print("=" * 50)
    
    for i, text in enumerate(texts, 1):
        print(f"\nText {i}:")
        print("-" * 10)
        results = pipeline(text)
        
        print(f"Summary: {results['summary']}")
        print(f"Sentiment: pos={results['sentiment']['positive']:.2f}, "
              f"neg={results['sentiment']['negative']:.2f}, "
              f"neu={results['sentiment']['neutral']:.2f}")
        print("Entities:", ", ".join(f"{e['name']} ({e['type']})" for e in results['entities']))
        print("=" * 50)
