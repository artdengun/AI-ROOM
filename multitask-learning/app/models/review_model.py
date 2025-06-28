from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20))
    emotion = db.Column(db.String(20))
    fakeness = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=db.func.now())