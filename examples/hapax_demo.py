"""
Hapax Demo: Type-Safe AI Pipelines

This example demonstrates Hapax's key features for building reliable AI pipelines:
1. Type safety at multiple stages
2. Graph-based pipeline construction
3. Evaluation frameworks
4. Integration with OpenLIT monitoring

The example implements a document processing pipeline that extracts information,
analyzes sentiment, and checks factual accuracy.
"""
import os
import time
import json
from typing import Dict, Any
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Mock OpenAI client for demonstration purposes
# In a real application, use the actual client:
# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
class MockOpenAIResponse:
    def __init__(self, content, tokens=10):
        self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content})()})]
        self.usage = type('obj', (object,), {'total_tokens': tokens})

class MockOpenAI:
    def chat_completions_create(self, model, messages, **kwargs):
        # Simulate LLM response based on the prompt
        prompt = messages[-1]["content"]
        if "extract entities" in prompt.lower() or "key information" in prompt.lower():
            return MockOpenAIResponse("Person: John Smith, Location: New York, Organization: Acme Corp", 15)
        elif "sentiment" in prompt.lower():
            return MockOpenAIResponse("Positive", 5)
        elif "fact check" in prompt.lower() or "accurate" in prompt.lower():
            return MockOpenAIResponse("The information appears to be accurate", 8)
        else:
            return MockOpenAIResponse("I'm not sure how to respond to that prompt", 7)

openai_client = MockOpenAI()

#############################################################################
# SECTION 1: STANDARD PYTHON IMPLEMENTATION
#############################################################################

def standard_implementation():
    """Demonstrate document processing with standard Python."""
    print("\n=== STANDARD PYTHON IMPLEMENTATION ===\n")
    
    # Define functions without type checking at definition time
    def extract_key_info(document):
        # No type checking here - could receive any type
        try:
            # Simple logging instead of OpenLIT instrumentation
            print("  Starting entity extraction...")
            
            # Type check is done at runtime - will fail for non-string inputs
            if not isinstance(document, str):
                raise TypeError(f"Expected document to be a string, got {type(document)}")
            
            start_time = time.time()
            response = openai_client.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract key entities (people, places, organizations) from the text."},
                    {"role": "user", "content": document}
                ]
            )
            
            # Print some metrics that would be recorded in OpenLIT
            processing_time = time.time() - start_time
            print(f"  Processing time: {processing_time:.2f}s, Tokens: {response.usage.total_tokens}")
            
            # Parse the response - could fail if response format is unexpected
            entities_text = response.choices[0].message.content
            print(f"  Extracted entities: {entities_text}")
            
            return entities_text
        except Exception as e:
            print(f"  Error in extract_key_info: {e}")
            # Propagate the error instead of returning a default value
            raise
    
    def analyze_sentiment(document):
        try:
            print("  Starting sentiment analysis...")
            
            # Type check is done at runtime - will fail for non-string inputs
            if not isinstance(document, str):
                raise TypeError(f"Expected document to be a string, got {type(document)}")
            
            start_time = time.time()
            response = openai_client.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Analyze the sentiment of this text as positive, negative, or neutral."},
                    {"role": "user", "content": document}
                ]
            )
            
            # Print some metrics that would be recorded in OpenLIT
            processing_time = time.time() - start_time
            print(f"  Processing time: {processing_time:.2f}s, Tokens: {response.usage.total_tokens}")
            
            sentiment = response.choices[0].message.content
            print(f"  Sentiment analysis: {sentiment}")
            
            return sentiment
        except Exception as e:
            print(f"  Error in analyze_sentiment: {e}")
            # Propagate the error instead of returning a default value
            raise
    
    def check_factual_accuracy(document, entities):
        try:
            print("  Starting factual accuracy check...")
            
            # Manual error checking - no automatic type validation
            if not isinstance(document, str):
                raise TypeError(f"Expected document to be a string, got {type(document)}")
            if not isinstance(entities, str):
                raise TypeError(f"Expected entities as string, got {type(entities)}")
            
            start_time = time.time()
            response = openai_client.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Check if the following information appears factually accurate."},
                    {"role": "user", "content": f"Text: {document}\nExtracted information: {entities}"}
                ]
            )
            
            # Print some metrics that would be recorded in OpenLIT
            processing_time = time.time() - start_time
            print(f"  Processing time: {processing_time:.2f}s, Tokens: {response.usage.total_tokens}")
            
            accuracy_assessment = response.choices[0].message.content
            print(f"  Factual accuracy: {accuracy_assessment}")
            
            return accuracy_assessment
        except Exception as e:
            print(f"  Error in check_factual_accuracy: {e}")
            # Propagate the error instead of returning a default value
            raise
    
    def combine_results(document, entities, sentiment, accuracy):
        try:
            print("  Combining results...")
            
            # Manual type checking and error handling
            if not isinstance(document, str):
                raise TypeError(f"Expected document to be a string, got {type(document)}")
            
            result = {
                "document_preview": document[:50] + "..." if len(document) > 50 else document,
                "entities": entities,
                "sentiment": sentiment,
                "accuracy_assessment": accuracy,
                "timestamp": time.time()
            }
            
            print(f"  Combined results: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            print(f"  Error in combine_results: {e}")
            # Propagate the error instead of returning a default value
            raise
    
    def process_document(document):
        """Process a document through the entire pipeline."""
        try:
            print("\nStarting document processing...")
            
            # Manual pipeline execution - prone to errors in sequencing
            
            # Step 1: Extract entities - type issues caught only at runtime
            entities = extract_key_info(document)
            
            # Step 2: Analyze sentiment
            sentiment = analyze_sentiment(document)
            
            # Step 3: Fact checking - parameter order matters and is error-prone
            accuracy = check_factual_accuracy(document, entities)
            
            # Step 4: Combine results
            result = combine_results(document, entities, sentiment, accuracy)
            
            print("Document processing completed successfully")
            return result
            
        except Exception as e:
            # Manual error handling and reporting
            print(f"Document processing failed: {e}")
            print(traceback.format_exc())
            
            return {"error": str(e), "success": False}
    
    # Demonstrate usage
    print("Processing document with standard Python implementation...")
    sample_text = "John Smith from Acme Corp announced a new partnership in New York yesterday. Customers are excited about the possibilities."
    
    try:
        # This will work fine
        result = process_document(sample_text)
        
        # Let's try with an incorrect input type
        print("\nTrying with incorrect input type (integer instead of string)...")
        try:
            result = process_document(42)  # Type error only caught at runtime
            print("  Result:", result)
        except Exception as e:
            print(f"  Runtime error caught: {e}")
    
    except Exception as e:
        print(f"Example failed: {e}")

#############################################################################
# SECTION 2: HAPAX IMPLEMENTATION
#############################################################################

def hapax_implementation():
    """Demonstrate document processing using Hapax."""
    print("\n=== HAPAX IMPLEMENTATION ===\n")
    
    try:
        from hapax import ops, Graph, set_openlit_config
        
        # Hapax automatically configures OpenLIT
        set_openlit_config({
            "otlp_endpoint": os.getenv("OPENLIT_ENDPOINT", "http://localhost:4318"),
            "application_name": "hapax_document_processor"
        })
        print("✓ Hapax initialized with OpenLIT integration")
        
        # TYPE CHECKING STAGE 1: Import-time validation with @ops decorator
        # This ensures the function has proper type annotations before it can be used
        @ops(
            name="extract_key_info",
            tags=["nlp", "extraction"]
        )
        def extract_key_info(document: str) -> str:
            """Extract key information from a document."""
            print("  Starting entity extraction...")
            
            response = openai_client.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract key entities (people, places, organizations) from the text."},
                    {"role": "user", "content": document}
                ]
            )
            
            entities_text = response.choices[0].message.content
            print(f"  Extracted entities: {entities_text}")
            return entities_text
        
        @ops(
            name="analyze_sentiment",
            tags=["nlp", "sentiment"]
        )
        def analyze_sentiment(document: str) -> str:
            """Analyze the sentiment of a document."""
            print("  Starting sentiment analysis...")
            
            response = openai_client.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Analyze the sentiment of this text as positive, negative, or neutral."},
                    {"role": "user", "content": document}
                ]
            )
            
            sentiment = response.choices[0].message.content
            print(f"  Sentiment analysis: {sentiment}")
            return sentiment
        
        @ops(
            name="check_factual_accuracy",
            tags=["nlp", "fact-checking"]
        )
        def check_factual_accuracy(document_and_entities: Dict[str, str]) -> str:
            """Evaluate the factual accuracy of extracted information."""
            print("  Starting factual accuracy check...")
            
            document = document_and_entities["document"]
            entities = document_and_entities["entities"]
            
            response = openai_client.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Check if the following information appears factually accurate."},
                    {"role": "user", "content": f"Text: {document}\nExtracted information: {entities}"}
                ]
            )
            
            accuracy_assessment = response.choices[0].message.content
            print(f"  Factual accuracy: {accuracy_assessment}")
            return accuracy_assessment
        
        @ops(
            name="combine_results",
            tags=["postprocessing"]
        )
        def combine_results(data: Dict[str, Any]) -> Dict[str, Any]:
            """Combine all analysis results into a single output."""
            print("  Combining results...")
            
            document = data["document"]
            entities = data["entities"]
            sentiment = data["sentiment"]
            accuracy = data["accuracy"]
            
            result = {
                "document_preview": document[:50] + "..." if len(document) > 50 else document,
                "entities": entities,
                "sentiment": sentiment,
                "accuracy_assessment": accuracy,
                "timestamp": time.time()
            }
            
            print(f"  Combined results: {json.dumps(result, indent=2)}")
            return result
        
        # Create a custom operation that processes data in steps without redundancy
        @ops(
            name="process_document",
            tags=["pipeline"]
        )
        def process_document_hapax(document: str) -> Dict[str, Any]:
            """Process a document through the Hapax pipeline."""
            
            # These functions are already decorated with @ops, so they'll have all the
            # monitoring and type checking built in
            entities = extract_key_info(document)
            sentiment = analyze_sentiment(document)
            
            # Prepare data for fact checking
            document_and_entities = {
                "document": document,
                "entities": entities
            }
            
            # Run fact checking
            accuracy = check_factual_accuracy(document_and_entities)
            
            # Combine all results
            data = {
                "document": document,
                "entities": entities,
                "sentiment": sentiment,
                "accuracy": accuracy
            }
            
            return combine_results(data)
        
        # TYPE CHECKING STAGE 2: Definition-time validation when building the graph
        # The graph structure ensures type compatibility between operations
        def create_document_pipeline() -> Graph:
            """Create a simple document processing pipeline using Hapax Graph API."""
            pipeline = Graph("document_processor", "Process documents with NLP analysis")
            pipeline.then(process_document_hapax)
            return pipeline
        
        # Demonstrate usage
        print("Processing document with Hapax...")
        sample_text = "John Smith from Acme Corp announced a new partnership in New York yesterday. Customers are excited about the possibilities."
        
        # Create and execute the pipeline
        try:
            # TYPE CHECKING STAGE 3: Runtime validation during execution
            pipeline = create_document_pipeline()
            result = pipeline.execute(sample_text)
            
            # Let's try with an incorrect input type
            print("\nTrying with incorrect input type (integer instead of string)...")
            try:
                # This will trigger the type checking in Hapax
                result = pipeline.execute(42)  # Type error caught during execution
                print("  Result:", result)
            except TypeError as e:
                print(f"  Type error caught: {e}")
            except Exception as e:
                print(f"  Other error caught: {e}")
        
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            print(traceback.format_exc())
        
    except ImportError:
        print("This example requires Hapax. Skipping the Hapax implementation.")

#############################################################################
# MAIN EXECUTION
#############################################################################

if __name__ == "__main__":
    print("=" * 80)
    print("                    HAPAX DEMO: TYPE-SAFE AI PIPELINES                  ")
    print("=" * 80)
    
    # First, demonstrate using standard Python
    standard_implementation()
    
    # Then, demonstrate using Hapax
    hapax_implementation()
    
    print("\n" + "=" * 80)
    print("KEY FEATURES OF HAPAX")
    print("=" * 80)
    print("""
1. TYPE SAFETY:
   - Standard Python: Type errors caught only at runtime
   - Hapax: Multi-stage type checking (import, definition, runtime)
   
2. PIPELINE CONSTRUCTION:
   - Standard Python: Manual function sequencing 
   - Hapax: Declarative graph-based pipeline definition
   
3. MONITORING INTEGRATION:
   - Standard Python: Manual instrumentation required
   - Hapax: Automatic monitoring with operation metadata
   
4. ERROR HANDLING:
   - Standard Python: Manual try/except in each function
   - Hapax: Centralized error handling with diagnostics
   
5. EVALUATION FRAMEWORK:
   - Standard Python: Custom implementation needed
   - Hapax: Built-in evaluators (@eval decorator)
    """)
    
    print("\nFor more examples, see the other files in the examples directory.") 