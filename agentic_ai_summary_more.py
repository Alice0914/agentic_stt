import json
import asyncio
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from langchain.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI

# === Sample JSON Script Data ===
script = {
  "call_id": "CALL_2024_001",
  "timestamp": "2024-06-04T10:30:00Z",
  "duration": "00:08:45",
  "participants": {
    "agent": {
      "name": "Sarah Johnson",
      "id": "AGENT_001",
      "department": "Customer Service"
    },
    "customer": {
      "name": "John Smith",
      "id": "CUST_789456",
      "phone": "+1-555-0123"
    }
  },
  "transcript": [
    {
      "speaker": "agent",
      "timestamp": "00:00:05",
      "text": "Good morning, thank you for calling TechSupport Solutions. This is Sarah, how can I help you today?"
    },
    {
      "speaker": "customer",
      "timestamp": "00:00:12",
      "text": "Hi Sarah, um, I'm having a really frustrating issue with my internet connection. It's been going on for like three days now and, you know, I work from home so this is really affecting my productivity."
    },
    {
      "speaker": "agent",
      "timestamp": "00:00:28",
      "text": "I'm sorry to hear about the internet issues you're experiencing. I completely understand how frustrating that must be, especially when you're working from home. Let me help you get this resolved today."
    },
    {
      "speaker": "customer",
      "timestamp": "00:00:42",
      "text": "Thank you, I really appreciate that. So basically, uh, the connection keeps dropping every few hours and sometimes it's really slow. Like, you know, pages take forever to load."
    },
    {
      "speaker": "agent",
      "timestamp": "00:00:58",
      "text": "I see. Can you tell me what type of internet plan you have with us and what equipment you're using? Also, are you experiencing this issue on all devices or just specific ones?"
    },
    {
      "speaker": "customer",
      "timestamp": "00:01:12",
      "text": "Um, I have the premium plan, the 100 megabit one. I'm using the router you guys provided last year. And yeah, it's happening on my laptop, my phone, everything that connects to the wifi."
    },
    {
      "speaker": "agent",
      "timestamp": "00:01:28",
      "text": "Thank you for that information. Let me check your account and run some diagnostics on your connection. I can see here that there have been some intermittent connectivity issues reported in your area."
    },
    {
      "speaker": "customer",
      "timestamp": "00:01:42",
      "text": "Oh really? So it's not just me? That's actually, you know, kind of a relief to hear."
    },
    {
      "speaker": "agent",
      "timestamp": "00:01:50",
      "text": "Yes, we've had a few reports from your neighborhood. Our technical team has been working on resolving the infrastructure issue. However, I'd also like to troubleshoot your specific setup to make sure everything is optimized on your end."
    },
    {
      "speaker": "customer",
      "timestamp": "00:02:08",
      "text": "Okay, that sounds good. What do you need me to do?"
    },
    {
      "speaker": "agent",
      "timestamp": "00:02:12",
      "text": "First, let's try restarting your router. Can you unplug it for about 30 seconds and then plug it back in? I'll stay on the line with you."
    },
    {
      "speaker": "customer",
      "timestamp": "00:02:22",
      "text": "Sure, let me do that now. Okay, I'm unplugging it... and now I'm plugging it back in. The lights are coming back on."
    },
    {
      "speaker": "agent",
      "timestamp": "00:02:45",
      "text": "Perfect. Now let's wait for all the lights to stabilize. While we're waiting, I'm going to schedule a technician visit for tomorrow to check the line quality and make sure there are no issues with the connection to your home."
    },
    {
      "speaker": "customer",
      "timestamp": "00:03:02",
      "text": "Oh wow, that's great service. I wasn't expecting you to send someone out so quickly. Thank you so much, Sarah."
    },
    {
      "speaker": "agent",
      "timestamp": "00:03:12",
      "text": "You're very welcome! I want to make sure we get this completely resolved for you. The technician will be there between 9 AM and 12 PM tomorrow. Is that time frame okay for you?"
    },
    {
      "speaker": "customer",
      "timestamp": "00:03:24",
      "text": "Yes, that works perfectly. I'll be working from home anyway. Um, is there anything else I should try in the meantime?"
    },
    {
      "speaker": "agent",
      "timestamp": "00:03:34",
      "text": "The restart should help for now. If you continue to experience issues, try moving closer to the router or using an ethernet cable for your most important work tasks. I'm also applying a service credit to your account for the inconvenience."
    },
    {
      "speaker": "customer",
      "timestamp": "00:03:52",
      "text": "Oh, you don't have to do that, but I really appreciate it. This has been such a positive experience. You've been so helpful and patient with me."
    },
    {
      "speaker": "agent",
      "timestamp": "00:04:05",
      "text": "I'm so glad I could help! Is there anything else I can assist you with today?"
    },
    {
      "speaker": "customer",
      "timestamp": "00:04:12",
      "text": "No, I think that covers everything. Thank you again, Sarah. You've made my day so much better."
    },
    {
      "speaker": "agent",
      "timestamp": "00:04:20",
      "text": "You're very welcome, John. Have a great day, and don't hesitate to call if you need any further assistance!"
    }
  ],
  "resolution": "Technician scheduled, service credit applied, temporary troubleshooting provided",
  "customer_satisfaction": "High",
  "tags": ["internet_issues", "technical_support", "positive_resolution", "technician_scheduled"]
}

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
    script_data: dict
    transcript: str
    clean_text: str
    summary: str
    sentiment: str
    note: str
    evaluation: str

# === 2. Script Parser Agent ===
def script_parser_agent(state) -> Annotated[dict, "transcript"]:
    """Parse script data and extract transcript text"""
    script_data = state["script_data"]
    
    try:
        # Extract transcript text from script structure
        transcript_parts = []
        for entry in script_data["transcript"]:
            speaker = entry["speaker"]
            text = entry["text"]
            transcript_parts.append(f"{speaker.capitalize()}: {text}")
        
        # Join all parts into a single transcript
        full_transcript = "\n".join(transcript_parts)
        
        return {"transcript": full_transcript}
    
    except KeyError as e:
        return {"transcript": f"Error: Missing key in script structure: {e}"}
    except Exception as e:
        return {"transcript": f"Error parsing script data: {e}"}

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
builder.add_node("ScriptParser", RunnableLambda(script_parser_agent))
builder.add_node("Preprocess", RunnableLambda(preprocess_agent))
builder.add_node("ParallelSummarySentiment", RunnableLambda(parallel_summary_sentiment_agent))
builder.add_node("NoteWriter", RunnableLambda(note_writer_agent))
builder.add_node("Evaluation", RunnableLambda(evaluation_agent))

# Define the workflow
builder.set_entry_point("ScriptParser")
builder.add_edge("ScriptParser", "Preprocess")
builder.add_edge("Preprocess", "ParallelSummarySentiment")
builder.add_edge("ParallelSummarySentiment", "NoteWriter")
builder.add_edge("NoteWriter", "Evaluation")
builder.set_finish_point("Evaluation")

# === 10. Execute Graph ===
def run_customer_service_analysis(script_data):
    """Run the complete customer service analysis pipeline"""
    graph = builder.compile()
    
    initial_state = {"script_data": script_data}
    output = graph.invoke(initial_state)
    
    return output

# === Alternative: Separate Parallel Agents in Graph ===
def create_parallel_graph():
    """Create a graph where Summary and Sentiment agents run in parallel branches"""
    builder_parallel = StateGraph(State)
    
    # Add all nodes
    builder_parallel.add_node("ScriptParser", RunnableLambda(script_parser_agent))
    builder_parallel.add_node("Preprocess", RunnableLambda(preprocess_agent))
    builder_parallel.add_node("Summary", RunnableLambda(summary_agent))
    builder_parallel.add_node("Sentiment", RunnableLambda(sentiment_agent))
    builder_parallel.add_node("NoteWriter", RunnableLambda(note_writer_agent))
    builder_parallel.add_node("Evaluation", RunnableLambda(evaluation_agent))
    
    # Define the workflow with parallel branches
    builder_parallel.set_entry_point("ScriptParser")
    builder_parallel.add_edge("ScriptParser", "Preprocess")
    
    # Create parallel branches after preprocessing
    builder_parallel.add_edge("Preprocess", "Summary")
    builder_parallel.add_edge("Preprocess", "Sentiment")
    
    # Both Summary and Sentiment feed into NoteWriter
    builder_parallel.add_edge("Summary", "NoteWriter")
    builder_parallel.add_edge("Sentiment", "NoteWriter")
    
    builder_parallel.add_edge("NoteWriter", "Evaluation")
    builder_parallel.set_finish_point("Evaluation")
    
    return builder_parallel.compile()

def run_parallel_analysis(script_data, use_separate_branches=False):
    """Run analysis with option for parallel processing"""
    if use_separate_branches:
        graph = create_parallel_graph()
        print("Using separate parallel branches for Summary and Sentiment analysis...")
    else:
        graph = builder.compile()
        print("Using async parallel processing within single agent...")
    
    initial_state = {"script_data": script_data}
    output = graph.invoke(initial_state)
    
    return output

# === Performance Comparison Function ===
import time

def compare_performance(script_data):
    """Compare performance between sequential and parallel processing"""
    print("="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Test parallel async processing
    start_time = time.time()
    result_parallel = run_parallel_analysis(script_data, use_separate_branches=False)
    parallel_time = time.time() - start_time
    
    # Test separate branch processing
    start_time = time.time()
    result_branches = run_parallel_analysis(script_data, use_separate_branches=True)
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
    
    try:
        # Run the analysis using the script variable
        result = run_customer_service_analysis(script)
        
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
        compare_performance(script)
        
    except Exception as e:
        print(f"Error running analysis: {e}")

# === 12. Additional Utility Functions ===
def load_script_metadata(script_data):
    """Load additional metadata from the script data"""
    try:
        metadata = {
            "call_id": script_data.get("call_id", "N/A"),
            "duration": script_data.get("duration", "N/A"),
            "agent_name": script_data.get("participants", {}).get("agent", {}).get("name", "N/A"),
            "customer_name": script_data.get("participants", {}).get("customer", {}).get("name", "N/A"),
            "resolution": script_data.get("resolution", "N/A"),
            "tags": script_data.get("tags", [])
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
def process_multiple_scripts(scripts_list):
    """Process multiple script objects"""
    results = []
    
    for i, script_data in enumerate(scripts_list):
        print(f"Processing script {i+1}/{len(scripts_list)}")
        try:
            result = run_customer_service_analysis(script_data)
            result["script_index"] = i
            results.append(result)
        except Exception as e:
            print(f"Error processing script {i}: {e}")
    
    return results

def create_summary_report(results_list):
    """Create a summary report from multiple analysis results"""
    if not results_list:
        return "No results to summarize"
    
    report_prompt = f"""Create a comprehensive summary report based on the following customer service analysis results:
    
    {json.dumps([{
        'script_index': r.get('script_index', 'unknown'),
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

# === Direct Processing Function (Alternative Usage) ===
def process_script_directly(script_data):
    """Process script data directly without using the graph (for testing)"""
    try:
        # Parse script
        parser_result = script_parser_agent({"script_data": script_data})
        
        # Preprocess
        preprocess_result = preprocess_agent(parser_result)
        
        # Get clean text
        clean_text = preprocess_result["clean_text"]
        
        # Run parallel analysis
        summary, sentiment = asyncio.run(parallel_summary_sentiment_async(clean_text))
        
        return {
            "transcript": parser_result["transcript"],
            "clean_text": clean_text,
            "summary": summary,
            "sentiment": sentiment
        }
    except Exception as e:
        return {"error": str(e)}