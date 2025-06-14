import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from langchain_ollama import OllamaLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    PARSER = "parser"
    SCREENER = "screener"
    BIAS_CHECKER = "bias_checker"
    SHORTLIST_GENERATOR = "shortlist_generator"
    MANAGER_HANDOFF = "manager_handoff"

@dataclass
class JobRequirements:
    job_title: str
    department: str
    required_skills: List[str]
    preferred_skills: List[str]
    min_experience: float
    max_experience: float
    education_requirements: List[str]
    location: str
    salary_range: str
    job_description: str

@dataclass
class CandidateProfile:
    candidate_id: str
    personal_info: Dict[str, str]
    summary: str
    experience: List[Dict[str, Any]]
    skills: Dict[str, List[str]]
    education: List[Dict[str, str]]
    certifications: List[str]
    projects: List[Dict[str, str]]
    total_experience_years: float
    raw_resume: str

@dataclass
class ScreeningResult:
    candidate_id: str
    overall_score: float
    dimension_scores: Dict[str, float]
    recommendation: str
    strengths: List[str]
    concerns: List[str]
    reasoning: str
    interview_focus_areas: List[str]

@dataclass
class BiasCheckResult:
    candidate_id: str
    bias_risk_level: str
    detected_issues: List[Dict[str, str]]
    compliance_status: str
    recommendations: List[str]
    alternative_evaluation: str

class BaseLLMAgent(ABC):
    """Base class for all LLM agents using LangChain Ollama"""
    
    def __init__(self, model_name: str = "llama2", ollama_url: str = "http://localhost:11434", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.ollama_url = ollama_url
        
        # Initialize LangChain Ollama LLM
        self.llm = OllamaLLM(
            base_url=ollama_url,
            model=model_name,
            temperature=temperature
        )
        
    def call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> str:
        """Make LLM call using LangChain Ollama"""
        try:
            # Combine system and user prompts
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            
            # Get response from LLM
            response = self.llm.invoke(full_prompt)
            return response
                    
        except Exception as e:
            logger.error(f"LangChain Ollama call failed: {str(e)}")
            raise
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data and return result"""
        pass

class ResumeParsingAgent(BaseLLMAgent):
    """Agent 1: Resume Collection & Parsing"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """
You are a Resume Parsing Specialist. Your task is to extract and structure information from resumes.

EXTRACT THE FOLLOWING:
- Personal Information: Name, email, phone, location
- Professional Summary: Brief overview of candidate
- Work Experience: Company, role, duration, key responsibilities
- Technical Skills: Programming languages, frameworks, tools
- Education: Degree, institution, graduation year
- Certifications: Professional certifications and licenses
- Projects: Notable projects with technologies used

OUTPUT FORMAT (Valid JSON only):
{
  "candidate_id": "unique_identifier",
  "personal_info": {
    "name": "",
    "email": "",
    "phone": "",
    "location": ""
  },
  "summary": "",
  "experience": [
    {
      "company": "",
      "role": "",
      "duration": "",
      "years_calculated": 0.0,
      "responsibilities": []
    }
  ],
  "skills": {
    "technical": [],
    "soft": [],
    "languages": []
  },
  "education": [
    {
      "degree": "",
      "institution": "",
      "year": "",
      "gpa": ""
    }
  ],
  "certifications": [],
  "projects": [],
  "total_experience_years": 0.0
}

PARSING RULES:
- Calculate total experience accurately
- Normalize skill names (e.g., "JS" → "JavaScript")
- Extract quantifiable achievements
- Generate unique candidate_id from email or name
- Return ONLY valid JSON, no additional text or explanations
- If information is missing, use empty strings or arrays
"""
    
    async def process(self, input_data: Dict[str, str]) -> CandidateProfile:
        """Parse resume and return structured candidate profile"""
        resume_text = input_data["resume_text"]
        
        user_prompt = f"""
Parse the following resume and extract structured information:

RESUME TEXT:
{resume_text}

Return the extracted information in the specified JSON format. Only return the JSON object, no other text.
"""
        
        try:
            response = self.call_llm(self.system_prompt, user_prompt)
            
            # Clean the response to extract only JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]
            
            # Find JSON object in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                response = response[start_idx:end_idx]
            
            parsed_data = json.loads(response)
            
            # Create CandidateProfile object
            profile = CandidateProfile(
                candidate_id=parsed_data.get("candidate_id", f"candidate_{datetime.now().timestamp()}"),
                personal_info=parsed_data.get("personal_info", {}),
                summary=parsed_data.get("summary", ""),
                experience=parsed_data.get("experience", []),
                skills=parsed_data.get("skills", {}),
                education=parsed_data.get("education", []),
                certifications=parsed_data.get("certifications", []),
                projects=parsed_data.get("projects", []),
                total_experience_years=parsed_data.get("total_experience_years", 0.0),
                raw_resume=resume_text
            )
            
            logger.info(f"Successfully parsed resume for candidate: {profile.candidate_id}")
            return profile
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Raw response: {response}")
            raise
        except Exception as e:
            logger.error(f"Resume parsing failed: {str(e)}")
            raise

class ScreeningAgent(BaseLLMAgent):
    """Agent 2: Screening & Shortlisting"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """
You are a Technical Recruiter AI specializing in candidate screening. Evaluate candidates against job requirements using multiple criteria.

SCREENING CRITERIA:
1. Technical Skills Match (40% weight)
2. Experience Relevance (30% weight)
3. Educational Qualifications (15% weight)
4. Cultural & Role Fit (15% weight)

SCORING SYSTEM:
- Excellent Match (90-100): Perfect fit
- Strong Match (75-89): Very good candidate
- Good Match (60-74): Qualified
- Fair Match (40-59): Some gaps
- Poor Match (0-39): Not qualified

OUTPUT FORMAT (Valid JSON only):
{
  "overall_score": 85,
  "dimension_scores": {
    "technical_skills": 90,
    "experience": 80,
    "education": 85,
    "cultural_fit": 75
  },
  "recommendation": "SHORTLIST",
  "strengths": ["Strong Python skills", "Relevant experience"],
  "concerns": ["Limited cloud experience"],
  "reasoning": "Detailed explanation of scoring decision",
  "interview_focus_areas": ["System design", "Team collaboration"]
}

Return ONLY valid JSON, no additional text or explanations.
"""
    
    async def process(self, input_data: Dict[str, Any]) -> ScreeningResult:
        """Screen candidate against job requirements"""
        candidate_profile = input_data["candidate_profile"]
        job_requirements = input_data["job_requirements"]
        
        user_prompt = f"""
Evaluate the following candidate against the job requirements:

JOB REQUIREMENTS:
- Title: {job_requirements.job_title}
- Required Skills: {', '.join(job_requirements.required_skills)}
- Preferred Skills: {', '.join(job_requirements.preferred_skills)}
- Experience: {job_requirements.min_experience}-{job_requirements.max_experience} years

CANDIDATE PROFILE:
- Name: {candidate_profile.personal_info.get('name', 'N/A')}
- Summary: {candidate_profile.summary}
- Total Experience: {candidate_profile.total_experience_years} years
- Technical Skills: {', '.join(candidate_profile.skills.get('technical', []))}

Provide screening evaluation in the specified JSON format. Only return the JSON object, no other text.
"""
        
        try:
            response = self.call_llm(self.system_prompt, user_prompt)
            
            # Clean the response to extract only JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]
            
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                response = response[start_idx:end_idx]
            
            screening_data = json.loads(response)
            
            result = ScreeningResult(
                candidate_id=candidate_profile.candidate_id,
                overall_score=screening_data.get("overall_score", 0),
                dimension_scores=screening_data.get("dimension_scores", {}),
                recommendation=screening_data.get("recommendation", "REJECT"),
                strengths=screening_data.get("strengths", []),
                concerns=screening_data.get("concerns", []),
                reasoning=screening_data.get("reasoning", ""),
                interview_focus_areas=screening_data.get("interview_focus_areas", [])
            )
            
            logger.info(f"Screening completed for candidate: {candidate_profile.candidate_id}, Score: {result.overall_score}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse screening JSON response: {str(e)}")
            logger.error(f"Raw response: {response}")
            raise
        except Exception as e:
            logger.error(f"Screening failed: {str(e)}")
            raise

class BiasCheckAgent(BaseLLMAgent):
    """Agent 3: Bias & Compliance Checks"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """
You are a Diversity & Compliance Officer AI. Ensure fair and legally compliant hiring practices.

BIAS DETECTION CATEGORIES:
1. Demographic Bias
2. Experience Bias
3. Language & Presentation Bias

OUTPUT FORMAT (Valid JSON only):
{
  "bias_risk_level": "LOW",
  "detected_issues": [
    {
      "category": "demographic_bias",
      "description": "Potential issue description",
      "evidence": "Evidence from evaluation",
      "severity": "LOW"
    }
  ],
  "compliance_status": "COMPLIANT",
  "recommendations": [
    "Focus on job-relevant skills only"
  ],
  "alternative_evaluation": "Revised assessment focusing on objective criteria"
}

Return ONLY valid JSON, no additional text or explanations.
"""
    
    async def process(self, input_data: Dict[str, Any]) -> BiasCheckResult:
        """Check for bias and compliance issues"""
        candidate_profile = input_data["candidate_profile"]
        screening_result = input_data["screening_result"]
        
        user_prompt = f"""
Review the following screening decision for potential bias and compliance issues:

CANDIDATE INFORMATION:
- Name: {candidate_profile.personal_info.get('name', 'N/A')}
- Experience: {candidate_profile.total_experience_years} years

SCREENING DECISION:
- Overall Score: {screening_result.overall_score}
- Recommendation: {screening_result.recommendation}
- Reasoning: {screening_result.reasoning}

Analyze for bias and compliance issues. Only return the JSON object, no other text.
"""
        
        try:
            response = self.call_llm(self.system_prompt, user_prompt)
            
            # Clean the response to extract only JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]
            
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                response = response[start_idx:end_idx]
            
            bias_data = json.loads(response)
            
            result = BiasCheckResult(
                candidate_id=candidate_profile.candidate_id,
                bias_risk_level=bias_data.get("bias_risk_level", "MEDIUM"),
                detected_issues=bias_data.get("detected_issues", []),
                compliance_status=bias_data.get("compliance_status", "COMPLIANT"),
                recommendations=bias_data.get("recommendations", []),
                alternative_evaluation=bias_data.get("alternative_evaluation", "")
            )
            
            logger.info(f"Bias check completed for candidate: {candidate_profile.candidate_id}, Risk: {result.bias_risk_level}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse bias check JSON response: {str(e)}")
            logger.error(f"Raw response: {response}")
            raise
        except Exception as e:
            logger.error(f"Bias check failed: {str(e)}")
            raise

class ShortlistGeneratorAgent(BaseLLMAgent):
    """Agent 4: Shortlist Generation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """
You are a Strategic Hiring Manager AI. Create optimized shortlists.

OUTPUT FORMAT (Valid JSON only):
{
  "shortlist": [
    {
      "candidate_id": "C001",
      "rank": 1,
      "overall_score": 92,
      "selection_rationale": "Top technical skills + experience",
      "interview_priority": "HIGH",
      "estimated_success_probability": 0.85
    }
  ],
  "shortlist_summary": {
    "total_candidates": 4,
    "diversity_score": 0.75,
    "average_technical_score": 87,
    "skill_coverage": ["Python", "AWS"],
    "risk_assessment": "LOW"
  },
  "interview_strategy": {
    "recommended_order": [1, 2, 3],
    "focus_areas_per_candidate": {},
    "panel_composition_suggestions": []
  },
  "backup_candidates": []
}

Return ONLY valid JSON, no additional text or explanations.
"""
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized shortlist"""
        candidates_data = input_data["candidates_data"]
        job_requirements = input_data["job_requirements"]
        
        candidates_summary = []
        for profile, screening, bias_check in candidates_data:
            candidates_summary.append({
                "candidate_id": profile.candidate_id,
                "name": profile.personal_info.get('name', 'N/A'),
                "overall_score": screening.overall_score,
                "recommendation": screening.recommendation,
                "bias_risk": bias_check.bias_risk_level,
                "experience_years": profile.total_experience_years
            })
        
        user_prompt = f"""
Generate an optimized shortlist from the candidates for: {job_requirements.job_title}

CANDIDATES:
{json.dumps(candidates_summary, indent=2)}

Create a balanced shortlist. Only return the JSON object, no other text.
"""
        
        try:
            response = self.call_llm(self.system_prompt, user_prompt)
            
            # Clean the response to extract only JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]
            
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                response = response[start_idx:end_idx]
            
            shortlist_data = json.loads(response)
            
            logger.info(f"Shortlist generated with {len(shortlist_data.get('shortlist', []))} candidates")
            return shortlist_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse shortlist JSON response: {str(e)}")
            logger.error(f"Raw response: {response}")
            raise
        except Exception as e:
            logger.error(f"Shortlist generation failed: {str(e)}")
            raise

class ManagerHandoffAgent(BaseLLMAgent):
    """Agent 5: Manager Handoff"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """
You are an Executive Assistant AI. Prepare briefing materials for hiring managers.

OUTPUT FORMAT (Valid JSON only):
{
  "executive_summary": {
    "shortlist_quality": "EXCELLENT",
    "top_recommendation": "candidate_id",
    "key_insights": [],
    "timeline_status": "ON_TRACK",
    "next_actions": []
  },
  "candidate_briefings": [
    {
      "candidate_id": "C001",
      "executive_summary": "Brief overview",
      "interview_focus": ["technical depth"],
      "questions_to_ask": [],
      "red_flags_to_watch": [],
      "reference_priorities": [],
      "offer_considerations": {
        "salary_expectation": "120-140k",
        "start_date_flexibility": "High",
        "relocation_needed": false
      }
    }
  ],
  "process_integrity": {
    "bias_checks_passed": true,
    "compliance_verified": true,
    "quality_score": 0.92,
    "screening_confidence": "HIGH"
  },
  "recommendations": {
    "interview_panel_composition": [],
    "interview_sequence": [],
    "decision_timeline": "2 weeks",
    "backup_plan": "Next tier candidates ready"
  }
}

Return ONLY valid JSON, no additional text or explanations.
"""
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate manager handoff package"""
        shortlist_data = input_data["shortlist_data"]
        job_requirements = input_data["job_requirements"]
        
        user_prompt = f"""
Prepare a manager handoff package for: {job_requirements.job_title}

SHORTLIST SUMMARY:
{json.dumps(shortlist_data.get('shortlist_summary', {}), indent=2)}

Generate complete manager handoff package. Only return the JSON object, no other text.
"""
        
        try:
            response = self.call_llm(self.system_prompt, user_prompt)
            
            # Clean the response to extract only JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]
            
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                response = response[start_idx:end_idx]
            
            handoff_data = json.loads(response)
            
            logger.info("Manager handoff package generated successfully")
            return handoff_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse handoff JSON response: {str(e)}")
            logger.error(f"Raw response: {response}")
            raise
        except Exception as e:
            logger.error(f"Manager handoff generation failed: {str(e)}")
            raise

class ResumeScreeningPipeline:
    """Main pipeline orchestrating all agents"""
    
    def __init__(self, model_name: str = "llama2", ollama_url: str = "http://localhost:11434"):
        self.parser_agent = ResumeParsingAgent(model_name=model_name, ollama_url=ollama_url)
        self.screening_agent = ScreeningAgent(model_name=model_name, ollama_url=ollama_url)
        self.bias_check_agent = BiasCheckAgent(model_name=model_name, ollama_url=ollama_url)
        self.shortlist_agent = ShortlistGeneratorAgent(model_name=model_name, ollama_url=ollama_url)
        self.handoff_agent = ManagerHandoffAgent(model_name=model_name, ollama_url=ollama_url)
        
        self.processed_candidates = []
    
    async def process_single_candidate(self, resume_text: str, job_requirements: JobRequirements) -> Dict[str, Any]:
        """Process a single candidate through the pipeline"""
        try:
            # Step 1: Parse Resume
            logger.info("Step 1: Parsing resume...")
            candidate_profile = await self.parser_agent.process({"resume_text": resume_text})
            
            # Step 2: Screen Candidate
            logger.info("Step 2: Screening candidate...")
            screening_result = await self.screening_agent.process({
                "candidate_profile": candidate_profile,
                "job_requirements": job_requirements
            })
            
            # Step 3: Bias Check
            logger.info("Step 3: Checking for bias...")
            bias_check_result = await self.bias_check_agent.process({
                "candidate_profile": candidate_profile,
                "screening_result": screening_result
            })
            
            candidate_data = {
                "profile": candidate_profile,
                "screening": screening_result,
                "bias_check": bias_check_result
            }
            
            self.processed_candidates.append((candidate_profile, screening_result, bias_check_result))
            
            logger.info(f"Candidate {candidate_profile.candidate_id} processed successfully")
            return candidate_data
            
        except Exception as e:
            logger.error(f"Failed to process candidate: {str(e)}")
            raise
    
    async def process_multiple_candidates(self, resumes_data: List[str], job_requirements: JobRequirements) -> Dict[str, Any]:
        """Process multiple candidates and generate shortlist"""
        try:
            # Process all candidates
            logger.info(f"Processing {len(resumes_data)} candidates...")
            
            # Process candidates sequentially to avoid overloading Ollama
            successful_candidates = []
            for i, resume_text in enumerate(resumes_data):
                try:
                    logger.info(f"Processing candidate {i+1}/{len(resumes_data)}")
                    result = await self.process_single_candidate(resume_text, job_requirements)
                    successful_candidates.append(result)
                except Exception as e:
                    logger.error(f"Failed to process candidate {i+1}: {str(e)}")
                    continue
            
            if not successful_candidates:
                raise ValueError("No candidates were successfully processed")
            
            # Step 4: Generate Shortlist
            logger.info("Step 4: Generating shortlist...")
            shortlist_data = await self.shortlist_agent.process({
                "candidates_data": self.processed_candidates,
                "job_requirements": job_requirements
            })
            
            # Step 5: Manager Handoff
            logger.info("Step 5: Preparing manager handoff...")
            handoff_data = await self.handoff_agent.process({
                "shortlist_data": shortlist_data,
                "candidates_data": self.processed_candidates,
                "job_requirements": job_requirements
            })
            
            final_result = {
                "candidates_processed": len(successful_candidates),
                "shortlist": shortlist_data,
                "manager_handoff": handoff_data,
                "individual_candidates": successful_candidates
            }
            
            logger.info("Resume screening pipeline completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            raise

# Usage Example
async def main():
    """Example usage of the Resume Screening Pipeline with LangChain Ollama"""
    
    # Initialize pipeline with LangChain Ollama
    pipeline = ResumeScreeningPipeline(
        model_name="mistral",  # You can change this to any model you have in Ollama
        ollama_url="http://localhost:11434"
    )
    
    # Define job requirements
    job_requirements = JobRequirements(
        job_title="Senior Backend Developer",
        department="Engineering",
        required_skills=["Python", "Django", "PostgreSQL", "REST APIs"],
        preferred_skills=["AWS", "Docker", "Redis", "GraphQL"],
        min_experience=3.0,
        max_experience=8.0,
        education_requirements=["Bachelor's in Computer Science", "Bachelor's in Engineering"],
        location="Remote",
        salary_range="$120,000 - $150,000",
        job_description="We are looking for a senior backend developer..."
    )
    
    # Sample resume texts
    resumes = [
        """
        John Doe
        Email: john.doe@email.com
        Phone: (555) 123-4567
        
        PROFESSIONAL SUMMARY
        Senior Backend Developer with 5 years of experience in Python and Django...
        
        EXPERIENCE
        Senior Developer at TechCorp (2020-2024)
        - Developed REST APIs using Django and PostgreSQL
        - Implemented caching with Redis
        - Deployed applications on AWS
        
        SKILLS
        Python, Django, PostgreSQL, AWS, Docker, Redis
        
        EDUCATION
        Bachelor of Science in Computer Science
        University of Technology, 2019
        """,
        
        """
        Jane Smith
        Email: jane.smith@email.com
        Phone: (555) 987-6543
        
        PROFESSIONAL SUMMARY
        Full-stack developer with 4 years of experience...
        
        EXPERIENCE
        Backend Developer at StartupXYZ (2021-2024)
        - Built scalable APIs with Python and Flask
        - Worked with PostgreSQL and MongoDB
        - Experience with GraphQL
        
        SKILLS
        Python, Flask, PostgreSQL, MongoDB, GraphQL, JavaScript
        
        EDUCATION
        Bachelor of Engineering in Software Engineering
        State University, 2020
        """
    ]
    
    try:
        # Process all candidates
        results = await pipeline.process_multiple_candidates(resumes, job_requirements)
        
        # Print results
        print("=== RESUME SCREENING RESULTS ===")
        print(f"Candidates Processed: {results['candidates_processed']}")
        print(f"Shortlisted Candidates: {len(results['shortlist']['shortlist'])}")
        
        print("\n=== SHORTLIST ===")
        for candidate in results['shortlist']['shortlist']:
            print(f"Rank {candidate['rank']}: {candidate['candidate_id']} (Score: {candidate['overall_score']})")
            print(f"  Rationale: {candidate['selection_rationale']}")
        
        print("\n=== MANAGER RECOMMENDATIONS ===")
        exec_summary = results['manager_handoff']['executive_summary']
        print(f"Shortlist Quality: {exec_summary['shortlist_quality']}")
        print(f"Top Recommendation: {exec_summary['top_recommendation']}")
        print(f"Timeline Status: {exec_summary['timeline_status']}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())