from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

#account silindiğinde postlar da silinsin!!
class Account(db.Model):
    __tablename__ = 'account'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=True, nullable=False)
    followers = db.Column(db.Integer)
    posts = db.relationship('Post', backref='account', lazy=True)

class Post(db.Model):
    __tablename__ = 'post'
    post_id = db.Column(db.Integer, primary_key=True)
    account_id = db.Column(db.Integer, db.ForeignKey('account.id'), nullable=False)
    date = db.Column(db.DateTime)
    likes = db.Column(db.Integer)
    #comment_count = db.Column(db.Integer)
    caption = db.Column(db.Text)
    poi_extracted = db.Column(db.Boolean, default=False)    #to prevent running llm twice on same post
    location = db.relationship('Location', backref='post', uselist=False)
    pois = db.relationship('ExtractedPOI', backref='post', lazy=True)

class Location(db.Model):
    __tablename__ = 'location'
    post_id = db.Column(db.Integer, db.ForeignKey('post.post_id'), primary_key=True)
    loc_url = db.Column(db.Text)
    loc_name = db.Column(db.Text)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    city = db.Column(db.Text)

class ExtractedPOI(db.Model):
    __tablename__ = 'extractedpois'
    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey('post.post_id'), nullable=False)
    poi_name = db.Column(db.String(100))
    poi_type = db.Column(db.String(50))
    poi_activity = db.Column(db.Text)
    count = db.Column(db.Integer, default=1)
    #score = db.Column(db.Float, default=0.0)