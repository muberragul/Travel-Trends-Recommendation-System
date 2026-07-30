# Başlatma:
# venv\Scripts\activate
# python app.py

# Tarayıcıda açma:
# http://127.0.0.1:5000/

from datetime import datetime, timedelta
import math
import csv
import io
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, make_response
from models import db, Account, Post, Location, ExtractedPOI
from sqlalchemy import func, desc
from dotenv import load_dotenv
import os
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / 'html5up-solid-state'

app = Flask(__name__, 
            template_folder=str(WEB_ROOT),
            static_folder=str(WEB_ROOT))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

date_threshold = datetime.now() - timedelta(days=365)

def impact_score(mentions, total_likes, avg_followers, most_recent_date):
        avg_followers = float(avg_followers or 1)
        total_likes = float(total_likes or 0)
        mentions = int(mentions or 0)
        # impact uses normalized engagement and mentions (log-scaled)
        engagement_rate = total_likes / avg_followers
        # logarithmic scaling to mentions (rewards frequency)
        mention_factor = 1.0 + math.log1p(mentions)
        base_score = engagement_rate * mention_factor

        # recency multiplier
        if most_recent_date:
            if isinstance(most_recent_date, datetime):
                most_recent_date = most_recent_date.date()    
            days_ago = (datetime.now().date() - most_recent_date).days
            
            if days_ago <= 30:
                recency_multiplier = 1.5  # hot trend
            elif days_ago <= 90:
                recency_multiplier = 1.3  # recent - trending
            elif days_ago <= 180:
                recency_multiplier = 1.1  # semi-recent
            else:
                recency_multiplier = 1.0  # old content
        else:
            recency_multiplier = 1.0    
        return base_score * recency_multiplier

@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory(str(WEB_ROOT / 'assets'), path)

@app.route('/images/<path:path>')
def send_images(path):
    return send_from_directory(str(WEB_ROOT / 'images'), path)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['GET'])
def query_pois():
    city = request.args.get('city')
    limit = request.args.get('limit', 10, type=int)
    offset = request.args.get('offset', 0, type=int)

    limit = min(limit, 50)
    
    if not city:
        return jsonify({"error": "missing city param"}), 400

    # Join Post -> Account -> Location, group by ExtractedPOI
    rows = (
        db.session.query(
            ExtractedPOI.poi_name.label("poi_name"),
            ExtractedPOI.poi_type.label("poi_type"),
            ExtractedPOI.poi_activity.label("poi_activity"),   
            func.count(ExtractedPOI.id).label("mentions"),
            func.sum(Post.likes).label("total_likes"),
            func.avg(Post.likes).label("avg_likes"),
            func.avg(Account.followers).label("avg_followers"),
            func.max(Post.date).label("most_recent_date"),
           )
        .join(Post, Post.post_id == ExtractedPOI.post_id)
        .join(Account, Account.id == Post.account_id)
        .join(Location, Location.post_id == Post.post_id)
        .filter(Location.city.ilike(f"%{city}%"))
        .filter(Post.date >= date_threshold)  # date range for getting recent posts
        .group_by(ExtractedPOI.poi_name, ExtractedPOI.poi_type, ExtractedPOI.poi_activity)
        .all()
    )

    results = []    
    for r in rows:
        normalized_poi_name = r.poi_name.strip().title() if r.poi_name else "N/A"
        sc = impact_score(r.mentions, r.total_likes, r.avg_followers, r.most_recent_date)
        results.append({
            "poi_name": normalized_poi_name,
            "poi_type": r.poi_type or "other",
            "poi_activity": r.poi_activity or "N/A",
            "mentions": int(r.mentions),
            "total_likes": int(r.total_likes or 0),
            "avg_likes": float(r.avg_likes or 0.0),
            "avg_followers": float(r.avg_followers or 0.0),
            "impact": round(sc, 6),
        })

    results_sorted = sorted(results, key=lambda x: x["impact"], reverse=True)

    # pagination
    total_count = len(results_sorted)      
    paginated_results = results_sorted[offset:offset + limit]
    has_more = (offset + limit) < total_count

    return jsonify({
        "city": city,
        "top_pois": paginated_results,
        "total": total_count,
        "has_more": has_more,
        "offset": offset,
    })

@app.route('/download', methods=['POST'])
def download_csv():
    try:
        data = request.json
        city = data.get('city', 'unknown')
        pois = data.get('pois', [])
        
        if not pois:
            return jsonify({"error": "No data to download"}), 400
        
        output = io.StringIO()
        writer = csv.writer(output)     
        writer.writerow(['Rank', 'POI Name', 'Category', 'Activity', 'Mentions', 'Total Likes', 'Avg Likes', 'Impact Score'])
        for idx, poi in enumerate(pois, 1):
            writer.writerow([
                idx,
                poi.get('poi_name', ''),
                poi.get('poi_type', ''),
                poi.get('poi_activity', ''),
                poi.get('mentions', 0),
                poi.get('total_likes', 0),
                round(poi.get('avg_likes', 0), 1),
                round(poi.get('impact', 0), 2)
            ])
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=travel_trends_{city}.csv'
        
        return response
    except Exception as e:
        print(f"Error in download: {str(e)}")
        return jsonify({"error": str(e)}), 500

"""
@app.route('/admin')
def admin():
    return render_template('admin.html')
"""
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)