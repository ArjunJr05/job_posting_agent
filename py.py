from langchain_ollama import OllamaLLM
import json

# Your job data (you can save this as job_parameters.json or use it directly)
job_data = {
    "company": "urn:li:company:123456",
    "companyApplyUrl": "http://linkedin.com",
    "description": "We are looking for a passionate Software Engineer to design, develop and install software solutions. Software Engineer responsibilities include gathering user requirements, defining system functionality and writing code in various languages. Our ideal candidates are familiar with the software development life cycle (SDLC) from preliminary system analysis to tests and deployment.",
    "employmentStatus": "PART_TIME",
    "externalJobPostingId": "1234",
    "listedAt": 1440716666,
    "jobPostingOperationType": "CREATE",
    "title": "Software Engineer",
    "location": "San Francisco, CA",
    "workplaceTypes": ["hybrid"],
    "skills_required": ["Python", "Microservices", "Docker", "Kubernetes"],
    "experience_level": "Mid-Level",
    "responsibilities": ["Build scalable backend services", "Design APIs", "Write unit and integration tests"],
    "qualifications": ["Bachelor's in CS or related", "Experience with distributed systems"],
    "company_name": "WhisprNet",
    "salary_range": "$100K–$140K"
}

# Option 1: Load from JSON file (uncomment if you want to use a separate file)
# with open("job_parameters.json") as f:
#     job_data = json.load(f)

# Helper function to safely get fields with fallback
def get_field(data, key, default="Not specified"):
    return data.get(key, default)

# Map your JSON fields to the expected format
def map_job_data(job):
    return {
        "location": get_field(job, "location"),
        "salary": get_field(job, "salary_range"),
        "employment_type": get_field(job, "employmentStatus"),
        "duration": get_field(job, "duration", "Ongoing"),  # Not in your JSON, using default
        "skills": get_field(job, "skills_required", []),
        "experience": get_field(job, "experience_level"),
        "english_level": get_field(job, "english_level", "Professional"),  # Not in your JSON, using default
        "level": get_field(job, "experience_level"),
        "title": get_field(job, "title"),
        "company_name": get_field(job, "company_name"),
        "description": get_field(job, "description"),
        "responsibilities": get_field(job, "responsibilities", []),
        "qualifications": get_field(job, "qualifications", []),
        "workplace_types": get_field(job, "workplaceTypes", [])
    }

# Map the job data
mapped_job = map_job_data(job_data)

# Dynamically construct the prompt
prompt = f"""
You are a professional recruiter. Based on the job parameters below, write a formal, friendly, and engaging job description suitable for posting on LinkedIn and Indeed.

Structure the job post as follows:

1. Greeting and Opening — urgency of hiring, number of open roles, and job title.

2. Project Summary — Describe what the company and project are about, including its purpose, goals, or impact.
Company: {mapped_job["company_name"]}
Description: {mapped_job["description"]}

3. Job Conditions — Include:
- Location: {mapped_job["location"]}
- Salary: {mapped_job["salary"]}
- Work Mode: {mapped_job["employment_type"]} ({', '.join(mapped_job["workplace_types"])})
- Duration: {mapped_job["duration"]}
- Contract Type: Direct contract unless specified otherwise.

4. Tech Stack — {', '.join(mapped_job["skills"])}

5. Requirements:
- Experience: {mapped_job["experience"]}
- English Level: {mapped_job["english_level"]}
- Level: {mapped_job["level"]}

Additional Qualifications: {', '.join(mapped_job["qualifications"])}

Key Responsibilities: {', '.join(mapped_job["responsibilities"])}

6. Call to Action — Encourage candidates to apply, ask questions, and send their CVs and salary expectations.

Important: Don't use double star(**) or emoji characters in the response.
"""

# Initialize the LLM
llm = OllamaLLM(base_url="http://localhost:11434", model="mistral")

# Get response
response = llm.invoke(prompt)

# Print result
print("\n📄 Generated Job Description:\n")
print(response)

# Optional: Save the generated description to a file
with open("generated_job_description.txt", "w", encoding="utf-8") as f:
    f.write(response)
    
print("\n✅ Job description saved to 'generated_job_description.txt'")