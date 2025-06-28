from app.models.review_model import db, Review

def save_prediction(text, sentiment, emotion, fakeness):
    review = Review(
        text=text,
        sentiment=sentiment,
        emotion=emotion,
        fakeness=fakeness
    )
    db.session.add(review)
    db.session.commit()