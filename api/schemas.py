from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    f0: float = Field(..., description="Feature 0")
    f1: float = Field(..., description="Feature 1")
    f2: float = Field(..., description="Feature 2")
    f3: float = Field(..., description="Feature 3")
    f4: float = Field(..., description="Feature 4")
    f5: float = Field(..., description="Feature 5")
    f6: float = Field(..., description="Feature 6")
    f7: float = Field(..., description="Feature 7")
    f8: float = Field(..., description="Feature 8")
    f9: float = Field(..., description="Feature 9")
    f10: float = Field(..., description="Feature 10")
    f11: float = Field(..., description="Feature 11")

class PredictResponse(BaseModel):
    uplift_score: float
    decision: str
    bid_eligible: bool
