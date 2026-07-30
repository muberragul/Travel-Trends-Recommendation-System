import os
import json
from typing import List, Dict, Any

from sqlalchemy import func
from app import app
from openai import OpenAI
import spacy
from dotenv import load_dotenv
from models import Location, db, ExtractedPOI, Post, Account
from fuzzy_matching import fuzzy_match_and_merge_poi, merge_poi_counts

load_dotenv()

BATCH_SIZE = 50  # can be tuned based on performance and rate limits

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

QWEN_KEY = os.environ.get("QWEN_API_KEY")

# extracts POIs per post using Qwen LLM.
# arg -> text (str): The input text (Instagram caption).
# returns -> List of POI dicts as described in the prompt.
def extract_pois_with_qwen(text, location=None, model = "qwen-turbo"):
    client = OpenAI(
        api_key= QWEN_KEY,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    input_lines = [f'Caption: "{text}"']
    if location:
        input_lines.append(f"Location: {location}")

    POI_PROMPT_INSTRUCTIONS = f"""
    You are given a social media post caption and optional location for a travel account.
    Task: Extract all places of interest (POIs) and short activity suggestions from the text. 
    Discard food names as POIs and one word non-proper nouns such as cafe, coffee, bridge, park, castle, church, station if they don't have a related name before.
    Return a JSON array of objects. Each object must contain:
    - poi_name: the canonical short name of the place (string), not a dish name
    - poi_type: one of [museum, park, market, restaurant, cafe, bar, lake, palace, beach, viewpoint, shopping, other]
    - activity: short description (verb phrase) of what to do there (string)

    Examples:
    Input: "Chill afternoon at Englischer Garten — rent a bike, visit the Chinese Tower beer garden, and watch surfers at the Eisbach wave."
    Output:
    [
        {{"poi_name":"Englischer Garten","poi_type":"park","activity":"rent a bike"}},
        {{"poi_name":"Chinese Tower","poi_type":"bar","activity":"drink at the beer garden"}},
        {{"poi_name":"Eisbach","poi_type":"other","activity":"watch surfers at the wave"}}
    ]

    If the text doesn't mention any POI, return an empty array: [].
    Only return valid JSON — do not include extra text.

    {input_lines}
    Output:
    """.strip()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': 'You are a JSON extractor for POIs.'},
                {'role': 'user', 'content': POI_PROMPT_INSTRUCTIONS}
            ]
        )
        output = completion.choices[0].message.content

        pois = json.loads(output)
        if isinstance(pois, list):
            return pois
    except Exception as e:
        print(f"Error extracting POIs: {e}")
    return []


# naive Named Entity Recognition (NER)-based fallback
# Spacy library used for the NER but it wasn't successful and needed a lot of finetuning. So LLM is preferred.
def extract_with_spacy(text: str, loc_name: str = "") -> List[Dict[str, Any]]: 
    if not nlp:
        return []
    doc = nlp(text + " " + (loc_name or ""))
    pois = []
    for ent in doc.ents:
        if ent.label_ in ("FAC", "ORG", "GPE", "LOC"):
            pois.append({
                "poi_name": ent.text,
                "poi_type": "other"
            })
    # deduplicate by name
    seen = set()
    out = []
    for p in pois:
        key = p["poi_name"].lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out

# extract and store one ExtractedPOI row per (post, poi_name).
# skips deduplication if same (post_id, poi_name) already exists.
def extract_and_store_for_post(post: Post) -> dict:
    result = {
        'pois_added': 0,
        'pois_merged': 0,
        'pois_exact_match': 0,
        'llm_used': False,
        'spacy_used': False,
        'no_pois': False,
        'error': False
    }

    caption = post.caption or ""
    loc_name = post.location.loc_name if post.location else ""

    results = []
    if QWEN_KEY:
        results = extract_pois_with_qwen(caption, location=loc_name)
        if results:
            result['llm_used'] = True
        #time.sleep(0.2)
    if not results:
        results = extract_with_spacy(caption, loc_name)
        if results:
            result['spacy_used'] = True

    if not results:
        result['no_pois'] = True
        post.poi_extracted = True
        db.session.commit()
        return result

    for r in results:
        poi_name = (r.get("poi_name") or "").strip()
        if not poi_name:
            continue
        poi_type = (r.get("poi_type") or "other").strip()
        poi_activity = (r.get("activity") or "").strip()

        exists = ExtractedPOI.query.filter_by(post_id=post.post_id, poi_name=poi_name).first()
        if exists:
            continue

        # exact match in other posts (same city, case-insensitive)
        city = post.location.city if post.location else None
        if city:
            existing_exact_other_post = (
                db.session.query(ExtractedPOI).join(Post, Post.post_id == ExtractedPOI.post_id).join(Location, Location.post_id == Post.post_id)
                .filter(Location.city.ilike(f"%{city}%"),func.lower(ExtractedPOI.poi_name) == func.lower(poi_name),ExtractedPOI.poi_type == poi_type).first()
            )
            if existing_exact_other_post:
                merge_poi_counts(existing_exact_other_post, post, poi_activity)
                result['pois_exact_match'] += 1
                print(f"[EXACT] '{poi_name}' matched exactly - count incremented")
                continue

        # fuzzy matching with existing POIs in the same city
        matched_poi= fuzzy_match_and_merge_poi(
            post, poi_name, poi_type, poi_activity
        )
        if matched_poi:
            merge_poi_counts(matched_poi, post, poi_activity)
            result['pois_merged'] += 1
            print(f"[FUZZY] '{poi_name}' matched with '{matched_poi.poi_name}'")
        else:
            new = ExtractedPOI(post_id=post.post_id, poi_name=poi_name, poi_type=poi_type, poi_activity=poi_activity, count=1)
            db.session.add(new)
            result['pois_added'] += 1
    
    if result['pois_added'] > 0 or result['pois_merged'] > 0 or result['pois_exact_match'] > 0:
        post.poi_extracted = True
        db.session.commit()

    return result

def run_all(unprocessed_only: bool = True):
    stats = {
        'posts_processed': 0,
        'pois_extracted': 0,
        'posts_with_pois': 0,
        'posts_no_pois': 0,
        'llm_success': 0,
        'fallback_spacy': 0,
        'errors': 0
    }
    with app.app_context():
        if unprocessed_only:
            posts = Post.query.filter_by(poi_extracted=False).order_by(Post.date.desc()).all()
        else:
            posts = Post.query.order_by(Post.date.desc()).all()
        total_posts = len(posts)

        if total_posts == 0:
            print("[INFO] No posts to process (all posts already extracted)")
            return

        print(f"[INFO] Starting POI extraction for {total_posts} posts...\n")

        for i, post in enumerate(posts):
            if (i + 1) % 10 == 0 or (i + 1) == total_posts:
                print(f"[PROGRESS] {i+1}/{total_posts} posts processed...")
            try:
                result = extract_and_store_for_post(post)

                stats['posts_processed'] += 1
                stats['pois_extracted'] += result['pois_added']
                
                if result['pois_added'] > 0:
                    stats['posts_with_pois'] += 1
                
                if result['no_pois']:
                    stats['posts_no_pois'] += 1
                
                if result['llm_used']:
                    stats['llm_success'] += 1
                
                if result['spacy_used']:
                    stats['fallback_spacy'] += 1
            except Exception as e:
                print(f"[ERROR] Post {post.post_id}: {e}")
                stats['errors'] += 1
                db.session.rollback()
                continue

    print("\n" + "="*60)
    print("POI EXTRACTION SUMMARY")
    print("="*60)
    print(f"Posts processed:      {stats['posts_processed']}")
    print(f"Posts with POIs:      {stats['posts_with_pois']}")
    print(f"Posts without POIs:   {stats['posts_no_pois']}")
    print(f"Total POIs extracted: {stats['pois_extracted']}")
    print(f"LLM extractions:      {stats['llm_success']}")
    print(f"Fallback (spaCy):     {stats['fallback_spacy']}")
    print(f"Errors:               {stats['errors']}")
    print("="*60 + "\n")