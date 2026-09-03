from pydantic import BaseModel, Field


class Education(BaseModel):
    degree: str = ""
    institution: str = ""
    year: str = ""


class Project(BaseModel):
    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class Experience(BaseModel):
    company: str = ""
    role: str = ""
    start_date: str = ""
    end_date: str = ""
    description: str = ""


class CandidateProfile(BaseModel):
    name: str
    email: str
    education: list[Education] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    work_experience: list[Experience] = Field(default_factory=list)


def compact_candidate(candidate):
    if not candidate:
        return {}

    return {
        "skills": candidate.get("skills", [])[:15],

        "projects": [
            {
                "name": p.get("name", ""),
                "technologies": p.get("technologies", [])[:8],
                "description": p.get("description", "")[:500],
            }
            for p in candidate.get("projects", [])[:3]
        ],

        "work_experience": [
            {
                "company": exp.get("company", ""),
                "role": exp.get("role", ""),
                "description": exp.get("description", "")[:500],
            }
            for exp in candidate.get("work_experience", [])[:3]
        ],

        "education": [
            {
                "degree": edu.get("degree", ""),
                "institution": edu.get("institution", ""),
            }
            for edu in candidate.get("education", [])[:2]
        ],
    }