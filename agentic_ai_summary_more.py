script = {
            "callID": ["C1234"],
            "agentID": ["A1234"],
            "transcript": [
                {
                    "agent": "Good afternoon, thank you for calling XYZ Gas Company. How can I assist you today?",
                    "customer": "Hi, um I'm having an issue with my gas bill. It seems higher than usual this month."
                },
                {
                    "agent": "I'm sorry to hear that. Let's take a look at your account. Can you please provide your account number?",
                    "customer": "Sure, uh it's 123456789."
                },
                {
                    "agent": "Thank you. I see that your bill for this month is $150. Did you notice any unusual gas usage or changes in your household recently?",
                    "customer": "No, everything has been pretty normal. That's why I'm confused about the higher bill."
                },
                {
                    "agent": "I understand. Let me check your usage history and see if there are any anomalies. One moment, please. I see that your usage has increased by 20% compared to last month. This could be due to colder weather or a possible leak. Have you checked for any leaks in your gas appliances?",
                    "customer": "No, I haven't checked for leaks. How can I do that?"
                },
                {
                    "agent": "You can check for leaks by applying soapy water to the connections of your gas appliances. If you see bubbles, there might be a leak. Alternatively, we can schedule a technician to inspect your system.",
                    "customer": "I think I'll need a technician to come out and check. Can you schedule that for me?"
                },
                {
                    "agent": "Of course. I can schedule a technician to visit your home tomorrow between 10 AM and 12 PM. Does that work for you?",
                    "customer": "Yes, that works. Thank you."
                },
                {
                    "agent": "You're welcome. The technician will be there tomorrow morning. Is there anything else I can assist you with today?",
                    "customer": "Actually, yes. I need to update my contact information."
                },
                {
                    "agent": "Sure, I can help with that. What information would you like to update?",
                    "customer": "I'd like to update my phone number and email address."
                },
                {
                    "agent": "Alright, can you please provide the new phone number and email address?",
                    "customer": "My new phone number is 555-1234 and my new email address is example@example.com."
                },
                {
                    "agent": "Thank you. I've updated your contact information. Is there anything else I can assist you with?",
                    "customer": "Yes, I have a question about my recent payment."
                },
                {
                    "agent": "Sure, what is your question regarding the payment?",
                    "customer": "I made a payment last week, but it hasn't been reflected in my account yet."
                },
                {
                    "agent": "Let me check that for you. One moment, please. I see that your payment was received, but it hasn't been processed yet. It should be reflected in your account within the next 24 hours.",
                    "customer": "Okay, thank you for checking."
                },
                {
                    "agent": "You're welcome. Is there anything else I can assist you with?",
                    "customer": "No, that's all. Thanks for your help."
                },
                {
                    "agent": "My pleasure. Have a great day!",
                    "customer": "You too."
                }
            ]
}
# === 2. Preprocessing Agent ===
def preprocess_agent(script) -> Annotated[dict, "clean_text"]:
    transcript = script["transcript"]
    transcript_str = " ".join([f"{entry['agent']} {entry['customer']}" for entry in transcript])
    clean = transcript_str.replace("um", "").replace("uh", "").replace("you know", "")
    clean = " ".join(clean.split())
    return {"clean_text": clean}

clean_text = preprocess_agent(script)

# === 3. Summary Agent ===
async def summary_agent_async(clean_text: str) -> str:
    """Generate summary of the conversation"""
    prompt_summary = f"""please provide a concise summary of the following customer service conversation, 
                         focusing on the main issue, resolution steps taken, and outcome:
                         {clean_text}"""
    try:
        response = await llm.ainvoke(prompt_summary)
        return response.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

async def summary_agent(state) -> Annotated[dict, "summary"]:
    """Wrapper for summary agent"""
    clean_text = state["clean_text"]
    summary = await summary_agent_async(clean_text)
    return {"summary": summary}

# Generate the summary using the clean text
async def main():
    summary_output = await summary_agent(clean_text)
    print(summary_output)

# Run the async function
await main()

# === 5. Sentiment Agent (Separate) ===
async def sentiment_agent_async(clean_text: str) -> str:
    """Analyze sentiment of the conversation asynchronously"""
    prompt_sentiment = f"""Analyze the overall sentiment of this customer service conversation. 
    Consider both the customer's initial frustration and their final satisfaction level.
    Provide the sentiment as one of: Positive, Negative, or Neutral, followed by a brief explanation:
    
    {clean_text}"""
    
    try:
        response = await llm.ainvoke(prompt_sentiment)
        return response.content
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

async def sentiment_agent(state) -> Annotated[dict, "sentiment"]:
    """Wrapper for sentiment agent"""
    clean_text = state["clean_text"]
    sentiment = await sentiment_agent_async(clean_text)
    return {"sentiment": sentiment}
# Generate the sentiment using the clean text
async def main():
    sentiment_output = await sentiment_agent(clean_text)
    print(sentiment_output)

# Run the async function
await main()

# === 6. Note Writer Agent ===
async def note_writer_agent_async(summary: str, sentiment: str) -> str:
    """Generate a professional customer service note"""
    prompt = f"""Based on the following information, create a professional customer service note 
    that would be suitable for a CRM system. Include the key points, actions taken, and next steps:
    
    Summary: {summary}
    
    Sentiment Analysis: {sentiment}
    
    Format the note in a clear, professional manner suitable for other agents to reference."""
    
    try:
        response = await note_llm.ainvoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating note: {str(e)}"

async def note_writer_agent(state) -> Annotated[dict, "note"]:
    """Wrapper for note writer agent"""
    summary = state["summary"]
    sentiment = state["sentiment"]
    note = await note_writer_agent_async(summary, sentiment)
    return {"note": note}

# === 7. Complete Pipeline ===
async def complete_analysis():
    """Run complete analysis and generate final note"""
    
    # 1. Get clean text (already done above)
    clean_text_result = preprocess_agent(script)
    clean_text = clean_text_result["clean_text"]
    
    # 2. Generate summary and sentiment in parallel
    summary_task = summary_agent_async(clean_text)
    sentiment_task = sentiment_agent_async(clean_text)
    
    # Run both tasks concurrently
    summary, sentiment = await asyncio.gather(summary_task, sentiment_task)
    
    # 3. Create state for note generation
    state = {
        "clean_text": clean_text,
        "summary": summary,
        "sentiment": sentiment
    }
    
    # 4. Generate note
    note_result = await note_writer_agent(state)
    note = note_result["note"]
    
    # 5. Output all results
    print("="*60)
    print("CUSTOMER SERVICE ANALYSIS RESULTS")
    print("="*60)
    
    print("\n📋 CLEAN TEXT:")
    print("-" * 30)
    print(clean_text[:500] + "..." if len(clean_text) > 500 else clean_text)
    
    print("\n📝 SUMMARY:")
    print("-" * 30)
    print(summary)
    
    print("\n😊 SENTIMENT ANALYSIS:")
    print("-" * 30)
    print(sentiment)
    
    print("\n📄 CUSTOMER SERVICE NOTE:")
    print("-" * 30)
    print(note)
    
    return {
        "clean_text": clean_text,
        "summary": summary,
        "sentiment": sentiment,
        "note": note
    }

# === 8. Run Complete Analysis ===
# Execute the complete pipeline
result = await complete_analysis()

# === Alternative: Step by Step Execution ===
async def step_by_step_execution():
    """Step by step execution if you want to see each step"""
    
    print("Step 1: Preprocessing...")
    clean_text_result = preprocess_agent(script)
    clean_text = clean_text_result["clean_text"]
    print("✅ Clean text generated")
    
    print("\nStep 2: Generating summary...")
    summary = await summary_agent_async(clean_text)
    print("✅ Summary generated")
    
    print("\nStep 3: Analyzing sentiment...")
    sentiment = await sentiment_agent_async(clean_text)
    print("✅ Sentiment analyzed")
    
    print("\nStep 4: Creating customer service note...")
    note = await note_writer_agent_async(summary, sentiment)
    print("✅ Note generated")
    
    print("\n" + "="*60)
    print("FINAL CUSTOMER SERVICE NOTE")
    print("="*60)
    print(note)
    
    return note

# Uncomment below to run step by step
# final_note = await step_by_step_execution()

# === 9. Quick Note Generation (if you already have summary and sentiment) ===
async def quick_note_generation(summary_text, sentiment_text):
    """Quick function to generate note if you already have summary and sentiment"""
    note = await note_writer_agent_async(summary_text, sentiment_text)
    print("📄 CUSTOMER SERVICE NOTE:")
    print("-" * 40)
    print(note)
    return note

# Example usage if you want to use previously generated summary and sentiment:
# note_result = await quick_note_generation("your_summary_here", "your_sentiment_here")
