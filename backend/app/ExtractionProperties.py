from pydantic import BaseModel, Field, validator
from typing import Optional, List

class GeneralProperties(BaseModel):
    date: str = Field(description="Date of this document?")
    name: str = Field(description="Patient's name")
    organization: str = Field(description="Organization, such as a physician, hospital, or a lab?")

class VisitSummaries(GeneralProperties):
    diagnosis: Optional[str] = Field(description="Diagnosis")
    medications: str = Field(description="Correct name of medications")
    medicationsinstructions: str = Field(description="Detailed description of all correct name medications instructions?")
    instructions: str = Field(description="General Instructions")


class PrescriptionProperties(GeneralProperties):
    medications: str = Field(description="Bulletpoints of medications with instructions?")
    medicationsinstructions: str = Field(description="Bulletpoints of medications instructions?")

class InsuranceDocumentProperties(GeneralProperties):
    ad: str = Field(description="annual deductible")
    snf: str = Field(description="skilled nursing facility coverage and its cost") 
    cost: float = Field(description="cost for 10 physical therapy visits this year")
    diabetesCoverage: str = Field(description="summary of diabetes coverage")
    contact: str = Field(description="Contact number for patients and caregivers to call in case of complaints or questions.")

class DischargeSummaryProperties(GeneralProperties):
    pass 

class LabReportDataProperties(BaseModel):
    HbA1cValue: float = Field(description="Numerical latest value of HbA1c level")
    HbA1cUnit: str = Field(description="Unit of HbA1c level")
    HbA1cUL: float = Field(description="Upper Limit of HbA1c level")
    HbA1cLL: float = Field(description="Lower Limit of HbA1c level")
    BloodSugarValue: float = Field(description="Numerical latest value of Blood Sugar level")
    BloodSugarUnit: str = Field(description="Unit of Blood Sugar level")
    BloodSugarUL: float = Field(description="Upper Limit of Blood Sugar level")
    BloodSugarLL: float = Field(description="Lower Limit of Blood Sugar level")
    date: str = Field(description="Date of lab report in dd/mm/yyyy format")
