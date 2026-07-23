from pydantic import BaseModel, Field

class Education(BaseModel):
    degree: str = ""
    institution: str = ""
    year: str = ""

class Project(BaseModel):
    name: str = ""
    description: str = ""
    technologies: list[str] = []

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