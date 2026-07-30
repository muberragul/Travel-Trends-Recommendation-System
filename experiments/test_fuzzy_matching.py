import sys
import os
from pathlib import Path

from sqlalchemy import func
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DATA_DIR = PROJECT_ROOT / 'tests' / 'evaluation_data'

import pandas as pd
from app import app
from models import db, Post, Account, Location, ExtractedPOI
from extract_poi import extract_pois_with_qwen
from fuzzy_matching import fuzzy_match_and_merge_poi, merge_poi_counts
from datetime import datetime

def load_test_data(csv_path):
    """Load test CSV data into database"""
    df = pd.read_csv(csv_path)
    
    with app.app_context():
        # Clear existing test data - use integer IDs instead of LIKE
        # Clear test accounts by username patterns
        Account.query.filter(
            Account.username.in_([
                'sydney_traveler', 'aussie_explorer', 'harbor_views', 
                'nature_lover', 'city_wanderer', 'beach_life',
                'art_enthusiast', 'history_buff', 'architecture_fan', 'park_explorer'
            ])
        ).delete(synchronize_session=False)
        
        db.session.commit()
        
        # Insert test data - use unique integer IDs
        import random
        base_id = random.randint(900000, 999999)  # Random base to avoid conflicts
        
        for idx, row in df.iterrows():
            # Check if account already exists
            account = Account.query.filter_by(username=row['username']).first()
            
            if not account:
                # Create account
                account = Account(
                    username=row['username'],
                    followers=row['followers']
                )
                db.session.add(account)
                db.session.flush()
            
            # Create post with unique integer ID
            post_id = base_id + idx
            post = Post(
                post_id=post_id,
                account_id=account.id,
                caption=row['caption'],
                date=datetime.fromisoformat(row['date'].replace('Z', '+00:00')),
                likes=row['likes'],
                poi_extracted=False
            )
            db.session.add(post)
            
            # Create location - remove loc_id if it doesn't exist in your model
            location = Location(
                post_id=post.post_id,
                loc_name=row['location_name'],
                # loc_id=row['location_id'],  # Remove this line
                city='Sydney',
                latitude=row['latitude'],
                longitude=row['longitude']
            )
            db.session.add(location)
        
        db.session.commit()
        print(f"✓ Loaded {len(df)} test posts into database")
        print(f"✓ Post IDs range: {base_id} to {base_id + len(df) - 1}")
        
        return base_id, base_id + len(df) - 1

def test_fuzzy_matching(min_post_id, max_post_id):
    """Test fuzzy matching functionality"""
    with app.app_context():
        # Query posts by ID range instead of LIKE
        posts = Post.query.filter(
            Post.post_id.between(min_post_id, max_post_id)
        ).all()
        
        results = []
        fuzzy_matches = []
        exact_matches = []
        
        print("\n" + "="*80)
        print("FUZZY MATCHING TEST RESULTS")
        print("="*80 + "\n")
        
        for post in posts:
            print(f"\n📝 Post: {post.post_id}")
            print(f"   Caption: {post.caption[:60]}...")
            print(f"   Location: {post.location.loc_name if post.location else 'N/A'}")
            
            # Extract POIs using LLM
            caption = post.caption or ""
            location = post.location.loc_name if post.location else ""
            extracted_pois = extract_pois_with_qwen(caption, location=location)
            
            if not extracted_pois:
                print(f"   ⚠️  No POIs extracted")
                continue
            
            for poi_data in extracted_pois:
                poi_name = poi_data.get('poi_name', '').strip()
                poi_type = poi_data.get('poi_type', 'other').strip()
                poi_activity = poi_data.get('activity', '').strip()
                
                if not poi_name:
                    continue
                
                print(f"\n   🔍 Extracted POI: '{poi_name}' (Type: {poi_type})")
                
                # Check for exact match in same post first
                existing_exact_same_post = ExtractedPOI.query.filter_by(
                    post_id=post.post_id, 
                    poi_name=poi_name
                ).first()
                
                if existing_exact_same_post:
                    print(f"   ✓ Exact match found (same post) - skipping")
                    continue
                
                # Check for exact match across ALL posts in the same city (case-insensitive)
                existing_exact_other_post = (
                    db.session.query(ExtractedPOI)
                    .join(Post, Post.post_id == ExtractedPOI.post_id)
                    .join(Location, Location.post_id == Post.post_id)
                    .filter(
                        Location.city == 'Sydney',
                        func.lower(ExtractedPOI.poi_name) == func.lower(poi_name),
                        ExtractedPOI.poi_type == poi_type
                    )
                    .first()
                )
                
                if existing_exact_other_post:
                    print(f"   ✓✓ EXACT MATCH FOUND in another post!")
                    print(f"      Original POI: '{existing_exact_other_post.poi_name}'")
                    print(f"      Action: Incrementing count and updating date")
                    
                    exact_matches.append({
                        'poi_name': poi_name,
                        'original_poi_name': existing_exact_other_post.poi_name,
                        'poi_type': poi_type,
                        'original_post': existing_exact_other_post.post_id,
                        'new_post': post.post_id
                    })
                    
                    # Merge the POI
                    merge_poi_counts(existing_exact_other_post, post, poi_activity)
                    continue
                
                # Try fuzzy matching (only if no exact match found)
                matched_poi = fuzzy_match_and_merge_poi(
                    post, poi_name, poi_type, poi_activity
                )
                
                if matched_poi:
                    print(f"   🎯 FUZZY MATCH FOUND!")
                    print(f"      Matched with: '{matched_poi.poi_name}'")
                    print(f"      Action: Merging counts")
                    
                    # Get original post for the matched POI
                    original_post = db.session.get(Post, matched_poi.post_id)
                    
                    fuzzy_matches.append({
                        'new_poi_name': poi_name,
                        'matched_poi_name': matched_poi.poi_name,
                        'poi_type': poi_type,
                        'original_post': original_post.post_id if original_post else 'N/A',
                        'new_post': post.post_id
                    })
                    
                    # Merge the POI
                    merge_poi_counts(matched_poi, post, poi_activity)
                else:
                    print(f"   ➕ No match found - creating new POI")
                    
                    new_poi = ExtractedPOI(
                        post_id=post.post_id,
                        poi_name=poi_name,
                        poi_type=poi_type,
                        poi_activity=poi_activity,
                        count=1
                    )
                    db.session.add(new_poi)
                
                results.append({
                    'post_id': post.post_id,
                    'extracted_poi': poi_name,
                    'poi_type': poi_type,
                    'matched': matched_poi is not None,
                })
        
        db.session.commit()
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\nTotal posts processed: {len(posts)}")
        print(f"Total POI extractions: {len(results)}")
        print(f"Exact matches found: {len(exact_matches)}")
        print(f"Fuzzy matches found: {len(fuzzy_matches)}")
        
        if exact_matches:
            print("\n📍 EXACT MATCHES DETAIL:")
            for match in exact_matches:
                print(f"\n   '{match['poi_name']}' (exact match)")
                print(f"   Type: {match['poi_type']}")
                print(f"   Posts: {match['new_post']} merged into {match['original_post']}")
        
        if fuzzy_matches:
            print("\n📊 FUZZY MATCHES DETAIL:")
            for match in fuzzy_matches:
                print(f"\n   '{match['new_poi_name']}' → '{match['matched_poi_name']}'")
                print(f"   Type: {match['poi_type']}")
                print(f"   Posts: {match['new_post']} merged into {match['original_post']}")
        
        # Show final POI counts
        print("\n" + "="*80)
        print("FINAL POI DATABASE STATE")
        print("="*80)
        
        all_pois = ExtractedPOI.query.join(
            Post, Post.post_id == ExtractedPOI.post_id
        ).join(
            Location, Location.post_id == Post.post_id
        ).filter(
            Location.city == 'Sydney'
        ).all()
        
        print(f"\nTotal unique POIs in Sydney: {len(all_pois)}")
        for poi in all_pois:
            original_post = db.session.get(Post, poi.post_id)
            print(f"\n   POI: {poi.poi_name}")
            print(f"   Type: {poi.poi_type}")
            print(f"   Count: {poi.count}")
            print(f"   Most Recent Date: {original_post.date if original_post else 'N/A'}")
            print(f"   Activity: {poi.poi_activity}")

if __name__ == '__main__':
    csv_path = EVAL_DATA_DIR / 'test_sydney_fuzzy.csv'
    
    print("Loading test data...")
    min_id, max_id = load_test_data(csv_path)
    
    print("\nRunning fuzzy matching test...")
    test_fuzzy_matching(min_id, max_id)
    
    print("\n✅ Test complete!")