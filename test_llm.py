from llm_client import summarize_structured

sample_text = """
Artificial Intelligence is rapidly evolving with new models like GPT-4 and GPT-5.
These models are used for chat, summarization, and reasoning across domains.
They have strengths like fluency, but limitations in reasoning and factual accuracy.
"""

system_prompt = "You are a careful research assistant. Write structured summaries."
user_prompt = "Summarize the following passage in about 200 words with clear headings."

summary = summarize_structured(system_prompt, user_prompt, sample_text)

print("=== SUMMARY OUTPUT ===")
print(summary)
