from pydantic import BaseModel, Field, validator
from typing import Optional, List

class GeneralProperties(BaseModel):
    date: str = Field(description="Date of this document?")
    name: str = Field(description="Patient's name")
    organization: str = Field(description="Organization, such as a physician, hospital, or a lab?")
    followUpIns: Optional[str] = Field(description="Follow-up instructions")

class VisitSummaries(GeneralProperties):
    diagnosis: Optional[str] = Field(description="Description of diagnosis")
    medications: str = Field(description="Correct name of medications")
    medicationsinstructions: str = Field(description="Detailed description of correct name medications instructions?")
    instructions: str = Field(description="General Instructions")


class PrescriptionProperties(GeneralProperties):
    medications: str = Field(description="Bulletpoints of medications with instructions?")
    medicationsinstructions: str = Field(description="Bulletpoints of medications instructions?")

class InsuranceDocumentProperties(GeneralProperties):
    ad: str = Field(description="Answer to the annual deductible question")
    snf: str = Field(description="Answer to the skilled nursing facility covered and its cost question") 
    cost: str = Field(description="Answer to cost for 10 physical therapy visits this year question")
    diabetesCoverage: str = Field(description="Answer to summarize my diabetes coverage question")
    contact: str = Field(description="Contact number for patients and caregivers to call in case of complaints or questions.")

class DischargeSummaryProperties(GeneralProperties):
    pass 

class LabReportProperties(GeneralProperties):
    value: float = Field(description="Numerical value of lab test")
    date: str = Field(description="Date of lab report")