from langchain_ollama import OllamaLLM
import json
# Load job data from JSON
with open("job_parameters.json") as f:
    job = json.load(f)

# Helper function to safely get fields with fallback
def get_field(key, default="Not specified"):
    return job.get(key, default)

# Dynamically construct the prompt
prompt = f"""
You are a professional recruiter. Based on the job parameters below, write a formal, friendly, and engaging job description suitable for posting on LinkedIn and Indeed. Structure the job post as follows:

1. **Greeting and Opening** —  urgency of hiring, number of open roles, and job title.
2. **Project Summary** — Describe what the company and project are about, including its purpose, goals, or impact.
3. **Job Conditions** — Include:
   - Location: {get_field("location")}
   - Salary: {get_field("salary")}
   - Work Mode: {get_field("employment_type")}
   - Duration: {get_field("duration")}
   - Contract Type: Direct contract unless specified otherwise.
4. **Tech Stack** — {', '.join(get_field("skills", []))}
5. **Requirements**:
   - Experience: {get_field("experience")}
   - English Level: {get_field("english_level")}
   - Level: {get_field("level")}
6. **Call to Action** — Encourage candidates to apply, ask questions, and send their CVs and salary expectations.
Important: Dont't use double star(**)
"""

# Initialize the LLM
llm = OllamaLLM(base_url="http://localhost:11434", model="mistral")

# Get response
response = llm.invoke(prompt)

# Print result
print("\n📄 Generated Job Description:\n")
print(response)
