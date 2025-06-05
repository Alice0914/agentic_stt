import json
import asyncio
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from langchain.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI

# === Azure OpenAI Configuration ===
COMPLETION_TOKENS = 1000

# Initialize Azure ChatOpenAI
llm = AzureChatOpenAI(
    deployment_name=os.environ["GPT4o_DEPLOYMENT_NAME"], 
    temperature=0, 
    max_tokens=COMPLETION_TOKENS
)

# Separate LLM instances for different tasks with optimized settings
summary_llm = AzureChatOpenAI(
    deployment_name=os.environ["GPT4o_DEPLOYMENT_NAME"], 
    temperature=0.3, 
    max_tokens=500
)

sentiment_llm = AzureChatOpenAI(
    deployment_name=os.environ["GPT4o_DEPLOYMENT_NAME"], 
    temperature=0.1, 
    max_tokens=300
)

note_llm = AzureChatOpenAI(
    deployment_name=os.environ["GPT4o_DEPLOYMENT_NAME"], 
    temperature=0.2, 
    max_tokens=600
)

evaluation_llm = AzureChatOpenAI(
    deployment_name=os.environ["GPT4o_DEPLOYMENT_NAME"], 
    temperature=0.1, 
    max_tokens=400
)

# === 1. State Definition ===
class State(TypedDict):
    json_path: str
    transcript: str
    clean_text: str
    summary: str
    sentiment: str
    note: str
    evaluation: str

# === 2. JSON Reader Agent ===
def json_reader_agent(state) -> Annotated[dict, "transcript"]:
    """Reads JSON file and extracts transcript text"""
    json_path = state["json_path"]
    
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract transcript text from JSON structure
        transcript_parts = []
        for entry in data["transcript"]:
            speaker = entry["speaker"]
            text = entry["text"]
            transcript_parts.append(f"{speaker.capitalize()}: {text}")
        
        # Join all parts into a single transcript
        full_transcript = "\n".join(transcript_parts)
        
        return {"transcript": full_transcript}
    
    except FileNotFoundError:
        return {"transcript": "Error: JSON file not found"}
    except KeyError as e:
        return {"transcript": f"Error: Missing key in JSON structure: {e}"}
    except Exception as e:
        return {"transcript": f"Error reading JSON file: {e}"}

# === 3. Preprocessing Agent ===
def preprocess_agent(state) -> Annotated[dict, "clean_text"]:
    """Clean the transcript by removing filler words and extra spaces"""
    transcript = state["transcript"]
    
    # Remove common filler words and clean up text
    clean = transcript.replace("um", "").replace("uh", "").replace("you know", "")
    clean = clean.replace("like,", "").replace(", like", "")
    
    # Clean up extra spaces and line breaks
    clean = " ".join(clean.split())
    
    return {"clean_text": clean}

# === 4. Summary Agent (Separate) ===
async def summary_agent_async(clean_text: str) -> str:
    """Generate summary of the conversation asynchronously"""
    prompt_summary = f"""Please provide a concise summary of the following customer service conversation, 
    focusing on the main issue, resolution steps taken, and outcome:
    
    {clean_text}"""
    
    try:
        response = await summary_llm.ainvoke(prompt_summary)
        return response.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def summary_agent(state) -> Annotated[dict, "summary"]:
    """Wrapper for summary agent"""
    clean_text = state["clean_text"]
    summary = asyncio.run(summary_agent_async(clean_text))
    return {"summary": summary}

# === 5. Sentiment Agent (Separate) ===
async def sentiment_agent_async(clean_text: str) -> str:
    """Analyze sentiment of the conversation asynchronously"""
    prompt_sentiment = f"""Analyze the overall sentiment of this customer service conversation. 
    Consider both the customer's initial frustration and their final satisfaction level.
    Provide the sentiment as one of: Positive, Negative, or Neutral, followed by a brief explanation:
    
    {clean_text}"""
    
    try:
        response = await sentiment_llm.ainvoke(prompt_sentiment)
        return response.content
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

def sentiment_agent(state) -> Annotated[dict, "sentiment"]:
    """Wrapper for sentiment agent"""
    clean_text = state["clean_text"]
    sentiment = asyncio.run(sentiment_agent_async(clean_text))
    return {"sentiment": sentiment}

# === 6. Parallel Summary and Sentiment Agent ===
async def parallel_summary_sentiment_async(clean_text: str) -> tuple:
    """Run summary and sentiment analysis in parallel"""
    # Create tasks for parallel execution
    summary_task = summary_agent_async(clean_text)
    sentiment_task = sentiment_agent_async(clean_text)
    
    # Run both tasks concurrently
    summary, sentiment = await asyncio.gather(summary_task, sentiment_task)
    
    return summary, sentiment

def parallel_summary_sentiment_agent(state) -> Annotated[dict, "summary | sentiment"]:
    """Generate summary and analyze sentiment in parallel"""
    clean_text = state["clean_text"]
    
    # Run both analyses in parallel
    summary, sentiment = asyncio.run(parallel_summary_sentiment_async(clean_text))
    
    return {"summary": summary, "sentiment": sentiment}

# === 7. Note Writer Agent ===
def note_writer_agent(state) -> Annotated[dict, "note"]:
    """Generate a professional customer service note"""
    summary = state["summary"]
    sentiment = state["sentiment"]
    
    prompt = f"""Based on the following information, create a professional customer service note 
    that would be suitable for a CRM system. Include the key points, actions taken, and next steps:
    
    Summary: {summary}
    
    Sentiment Analysis: {sentiment}
    
    Format the note in a clear, professional manner suitable for other agents to reference."""
    
    try:
        response = note_llm.invoke(prompt)
        note = response.content
    except Exception as e:
        note = f"Error generating note: {str(e)}"
    
    return {"note": note}

# === 8. Evaluation Agent ===
def evaluation_agent(state) -> Annotated[dict, "evaluation"]:
    """Evaluate the accuracy and quality of the generated summary"""
    prompt = f"""Evaluate the accuracy and completeness of the following summary against the original conversation:
    
    Original Conversation:
    {state['clean_text']}
    
    Generated Summary:
    {state['summary']}
    
    Please rate the summary on a scale of 1-5 based on:
    1. Accuracy of information
    2. Completeness of key points
    3. Clarity and conciseness
    4. Professional tone
    
    Provide your rating and detailed reasoning."""
    
    try:
        response = evaluation_llm.invoke(prompt)
        evaluation = response.content
    except Exception as e:
        evaluation = f"Error generating evaluation: {str(e)}"
    
    return {"evaluation": evaluation}

# === 9. Construct LangGraph DAG ===
builder = StateGraph(State)

# Add nodes
builder.add_node("JSONReader", RunnableLambda(json_reader_agent))
builder.add_node("Preprocess", RunnableLambda(preprocess_agent))
builder.add_node("ParallelSummarySentiment", RunnableLambda(parallel_summary_sentiment_agent))
builder.add_node("NoteWriter", RunnableLambda(note_writer_agent))
builder.add_node("Evaluation", RunnableLambda(evaluation_agent))

# Define the workflow
builder.set_entry_point("JSONReader")
builder.add_edge("JSONReader", "Preprocess")
builder.add_edge("Preprocess", "ParallelSummarySentiment")
builder.add_edge("ParallelSummarySentiment", "NoteWriter")
builder.add_edge("NoteWriter", "Evaluation")
builder.set_finish_point("Evaluation")

# === 10. Execute Graph ===
def run_customer_service_analysis(json_file_path):
    """Run the complete customer service analysis pipeline"""
    graph = builder.compile()
    
    initial_state = {"json_path": json_file_path}
    output = graph.invoke(initial_state)
    
    return output

# === Alternative: Separate Parallel Agents in Graph ===
def create_parallel_graph():
    """Create a graph where Summary and Sentiment agents run in parallel branches"""
    builder_parallel = StateGraph(State)
    
    # Add all nodes
    builder_parallel.add_node("JSONReader", RunnableLambda(json_reader_agent))
    builder_parallel.add_node("Preprocess", RunnableLambda(preprocess_agent))
    builder_parallel.add_node("Summary", RunnableLambda(summary_agent))
    builder_parallel.add_node("Sentiment", RunnableLambda(sentiment_agent))
    builder_parallel.add_node("NoteWriter", RunnableLambda(note_writer_agent))
    builder_parallel.add_node("Evaluation", RunnableLambda(evaluation_agent))
    
    # Define the workflow with parallel branches
    builder_parallel.set_entry_point("JSONReader")
    builder_parallel.add_edge("JSONReader", "Preprocess")
    
    # Create parallel branches after preprocessing
    builder_parallel.add_edge("Preprocess", "Summary")
    builder_parallel.add_edge("Preprocess", "Sentiment")
    
    # Both Summary and Sentiment feed into NoteWriter
    builder_parallel.add_edge("Summary", "NoteWriter")
    builder_parallel.add_edge("Sentiment", "NoteWriter")
    
    builder_parallel.add_edge("NoteWriter", "Evaluation")
    builder_parallel.set_finish_point("Evaluation")
    
    return builder_parallel.compile()

def run_parallel_analysis(json_file_path, use_separate_branches=False):
    """Run analysis with option for parallel processing"""
    if use_separate_branches:
        graph = create_parallel_graph()
        print("Using separate parallel branches for Summary and Sentiment analysis...")
    else:
        graph = builder.compile()
        print("Using async parallel processing within single agent...")
    
    initial_state = {"json_path": json_file_path}
    output = graph.invoke(initial_state)
    
    return output

# === Performance Comparison Function ===
import time

def compare_performance(json_file_path):
    """Compare performance between sequential and parallel processing"""
    print("="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Test parallel async processing
    start_time = time.time()
    result_parallel = run_parallel_analysis(json_file_path, use_separate_branches=False)
    parallel_time = time.time() - start_time
    
    # Test separate branch processing
    start_time = time.time()
    result_branches = run_parallel_analysis(json_file_path, use_separate_branches=True)
    branches_time = time.time() - start_time
    
    print(f"\n⏱️  Async Parallel Processing Time: {parallel_time:.2f} seconds")
    print(f"⏱️  Separate Branches Time: {branches_time:.2f} seconds")
    print(f"⚡ Speed Improvement: {((max(parallel_time, branches_time) / min(parallel_time, branches_time) - 1) * 100):.1f}%")
    
    return result_parallel

# === 11. Main execution ===
if __name__ == "__main__":
    # Ensure environment variable is set
    if "GPT4o_DEPLOYMENT_NAME" not in os.environ:
        print("Error: Please set the GPT4o_DEPLOYMENT_NAME environment variable")
        exit(1)
    
    # Path to your JSON transcript file
    json_file_path = "customer_transcript.json"
    
    try:
        # Run the analysis
        result = run_customer_service_analysis(json_file_path)
        
        # Output results
        print("="*50)
        print("CUSTOMER SERVICE ANALYSIS RESULTS")
        print("="*50)
        
        print("\n📋 ORIGINAL TRANSCRIPT:")
        print("-" * 30)
        print(result["transcript"])
        
        print("\n🧹 CLEANED TEXT:")
        print("-" * 30)
        print(result["clean_text"])
        
        print("\n📝 SUMMARY:")
        print("-" * 30)
        print(result["summary"])
        
        print("\n😊 SENTIMENT ANALYSIS:")
        print("-" * 30)
        print(result["sentiment"])
        
        print("\n📄 CUSTOMER SERVICE NOTE:")
        print("-" * 30)
        print(result["note"])
        
        print("\n⭐ EVALUATION:")
        print("-" * 30)
        print(result["evaluation"])
        
        # Run performance comparison
        print("\n" + "="*50)
        print("PERFORMANCE ANALYSIS")
        print("="*50)
        compare_performance(json_file_path)
        
    except Exception as e:
        print(f"Error running analysis: {e}")

# === 12. Additional Utility Functions ===
def load_transcript_metadata(json_file_path):
    """Load additional metadata from the JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        metadata = {
            "call_id": data.get("call_id", "N/A"),
            "duration": data.get("duration", "N/A"),
            "agent_name": data.get("participants", {}).get("agent", {}).get("name", "N/A"),
            "customer_name": data.get("participants", {}).get("customer", {}).get("name", "N/A"),
            "resolution": data.get("resolution", "N/A"),
            "tags": data.get("tags", [])
        }
        
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {}

def save_analysis_results(results, output_file_path):
    """Save the analysis results to a JSON file"""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        print(f"Analysis results saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

# === Advanced Usage Examples ===
def batch_process_transcripts(transcript_folder_path):
    """Process multiple transcript files in a folder"""
    import glob
    
    json_files = glob.glob(f"{transcript_folder_path}/*.json")
    results = []
    
    for json_file in json_files:
        print(f"Processing: {json_file}")
        try:
            result = run_customer_service_analysis(json_file)
            result["file_name"] = json_file
            results.append(result)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return results

def create_summary_report(results_list):
    """Create a summary report from multiple analysis results"""
    if not results_list:
        return "No results to summarize"
    
    report_prompt = f"""Create a comprehensive summary report based on the following customer service analysis results:
    
    {json.dumps([{
        'file': r.get('file_name', 'unknown'),
        'summary': r.get('summary', ''),
        'sentiment': r.get('sentiment', '')
    } for r in results_list], indent=2)}
    
    Please provide:
    1. Overall trends in customer issues
    2. Sentiment analysis summary
    3. Common resolution patterns
    4. Recommendations for improvement
    """
    
    try:
        response = llm.invoke(report_prompt)
        return response.content
    except Exception as e:
        return f"Error creating summary report: {str(e)}"