from fastapi import HTTPException, Depends, APIRouter
from schemas import FeedbackIn, FeedbackOut
from models import Feedback
from sqlalchemy.orm import Session
from database import SessionLocal
from typing import List
from sentiment import classify

router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"]
)


def get_db():
    """
    Dependency function that provides a SQLAlchemy Session for each request.
    Ensures that the session is closed once the request is finished,
    preventing connection leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/", response_model=FeedbackOut)
def create_feedback(feedback_in: FeedbackIn, db: Session = Depends(get_db)):
    """
    Create a new feedback entry with sentiment classification.

    Parameters:
        feedback_in (FeedbackIn): The feedback text submitted by the user.
        db (Session): SQLAlchemy database session (injected).
    """
    sentiment_label = classify(feedback_in.text)
    new_feedback = Feedback(
        text=feedback_in.text,
        sentiment=sentiment_label
    )
    try:
        db.add(new_feedback)
        db.commit()
        db.refresh(new_feedback)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=400, detail="Could not save feedback into database") from exc
    return new_feedback


@router.get("/", response_model=List[FeedbackOut])
def get_all_feedback(db: Session = Depends(get_db)):
    """
    Retrieve all stored feedback entries.

    Parameters:
        db (Session): SQLAlchemy database session (injected).
    """
    return db.query(Feedback).all()
