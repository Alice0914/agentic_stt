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

