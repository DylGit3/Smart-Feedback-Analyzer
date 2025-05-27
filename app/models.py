from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime
from database import Base


class Feedback(Base):
    """
    ORM (Object-Relational mapping) model for storing user feedback with associated sentiment analysis.
    """
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
