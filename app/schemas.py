from datetime import datetime
from pydantic import BaseModel


class FeedbackIn(BaseModel):
    """
    Schema for incoming user feedback.
    """
    text: str


class FeedbackOut(BaseModel):
    """
    Schema for feedback returned by the API, including sentiment and timestamp.
    """
    class Config:
        orm_mode = True

    id: int
    text: str
    sentiment: str
    timestamp: datetime
