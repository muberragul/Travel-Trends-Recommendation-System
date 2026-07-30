import re
from langcodes import best_match
from rapidfuzz import fuzz
from sqlalchemy import and_, func
from models import db, ExtractedPOI, Post, Account, Location

def normalize_poi_name(name: str) -> str:
    name = name.strip().lower()

    name = name.replace('café', 'cafe')
    name = name.replace('cafés', 'cafes')
    
    # Remove common suffixes that don't add semantic value
    suffixes_to_remove = [
        r'\s+(park|gardens?|museum|palace|castle|bridge|tower|building|center|centre|square|station|street|road|avenue|cafe|cafes|restaurant|restaurants|bar|bars|pub|pubs|cathedral|casa)$'
    ]
    
    for suffix in suffixes_to_remove:
        name = re.sub(suffix, '', name, flags=re.IGNORECASE)
    
    # Remove articles
    name = re.sub(r'\b(the|a|an)\b', '', name)
    
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

# at least 60% of the shorter name's words should match
def is_substring_match(name1: str, name2: str) -> bool:
    name1_lower = name1.lower().strip()
    name2_lower = name2.lower().strip()
    
    if len(name1_lower) < 5 or len(name2_lower) < 5:
        return False
    
    # check if one is substring of the other
    if name1_lower in name2_lower or name2_lower in name1_lower:
        words1 = set(name1_lower.split())
        words2 = set(name2_lower.split())
        
        # if they share significant words, it's a match
        common_words = words1.intersection(words2)
        shorter_set = min(len(words1), len(words2))
        
        if len(common_words) >= max(2, shorter_set * 0.6):
            return True
    
    return False

def is_generic_poi(name: str) -> bool:
    name_lower = name.lower().strip()
    
    generic_terms = {
        'cafe', 'cafes', 'café', 'cafés', 'restaurant', 'restaurants',
        'bar', 'bars', 'pub', 'pubs', 'park', 'parks', 'garden', 'gardens',
        'museum', 'bridge', 'tower', 'church', 'station', 'beach',
        'market', 'shop', 'shopping', 'hotel', 'building', 'house'
    }
    
    words = name_lower.split()
    if len(words) == 1:
        if words[0] in generic_terms:
            return True
        return False
    
    # check for patterns like "park cafés" (generic + generic)
    if len(words) == 2:
        if all(word in generic_terms for word in words):
            return True
        
    # saves original cafes etc
    if len(words) >= 2:
        non_generic_words = [word for word in words if word not in generic_terms]
        if len(non_generic_words) >= 1:
            return False
        else:
            return True
    
    return False

# to avoid matching every cafe to cafes etc.
def has_meaningful_overlap(name1: str, name2: str) -> bool:
    generic_terms = {
        'cafe', 'cafes', 'café', 'cafés', 'restaurant', 'restaurants',
        'bar', 'bars', 'pub', 'pubs', 'park', 'parks', 'garden', 'gardens',
        'museum', 'bridge', 'tower', 'church', 'station', 'beach',
        'market', 'shop', 'shopping', 'hotel', 'building', 'house', 'the', 'a', 'an'
    }
    
    words1 = set(name1.lower().strip().split())
    words2 = set(name2.lower().strip().split())
    
    common_words = words1.intersection(words2)
    
    meaningful_common = common_words - generic_terms
    
    if len(meaningful_common) == 0:
        return False
    
    return True

# checks if a similar (similarity >= threshold) POI already exists for the same city using fuzzy matching.
def fuzzy_match_and_merge_poi(post: Post, poi_name: str, poi_type: str, poi_activity: str, 
                               similarity_threshold: float = 85.0) -> ExtractedPOI:
    if not post.location:
        return (None, 0)
    
    city = post.location.city
    if not city:
        return (None, 0)

    if is_generic_poi(poi_name):
        print(f"      [SKIP] '{poi_name}' is too generic - not storing")
        return None
    
    existing_pois = (
        db.session.query(ExtractedPOI)
        .join(Post, Post.post_id == ExtractedPOI.post_id)
        .join(Location, Location.post_id == Post.post_id)
        .filter(
            and_(
                Location.city.ilike(f"%{city}%"),
                ExtractedPOI.poi_type == poi_type,
                func.lower(ExtractedPOI.poi_name) != func.lower(poi_name)  # exclude exact matches
            )
        )
        .all()
    )
    
    poi_name_lower = poi_name.lower().strip()
    poi_name_normalized = normalize_poi_name(poi_name)
    best_match = None
    best_score = 0.0
    
    # try and find the best fuzzy match
    for existing_poi in existing_pois:
        if is_generic_poi(existing_poi.poi_name):
            continue

        if not has_meaningful_overlap(poi_name, existing_poi.poi_name):
            # pois only share generic words like "café"
            continue
        
        existing_name_lower = existing_poi.poi_name.lower().strip()
        existing_name_normalized = normalize_poi_name(existing_poi.poi_name)

        # substring matching ("Harbour Bridge" in "Sydney Harbour Bridge")
        if is_substring_match(poi_name, existing_poi.poi_name):
            print(f"      [SUBSTRING] '{poi_name}' is substring of '{existing_poi.poi_name}'")
            return existing_poi
        
        if poi_name_normalized and existing_name_normalized and len(poi_name_normalized) > 3:
            if poi_name_normalized == existing_name_normalized:
                print(f"      [NORMALIZED] '{poi_name}' matches '{existing_poi.poi_name}' after normalization")
                return existing_poi
            
        # different fuzzy matching methods to take the best score
        scores = []
        
        # token sort ratio - handles word order
        scores.append(fuzz.token_sort_ratio(poi_name_lower, existing_name_lower))
        
        # token set ratio - handles extra/missing words better
        scores.append(fuzz.token_set_ratio(poi_name_lower, existing_name_lower))
        
        # partial ratio - handles substring matches
        scores.append(fuzz.partial_ratio(poi_name_lower, existing_name_lower))
        
        # normalized names for additional scoring (just in case)
        if poi_name_normalized and existing_name_normalized:
            scores.append(fuzz.token_sort_ratio(poi_name_normalized, existing_name_normalized))
        
        max_similarity = max(scores)
        
        if max_similarity >= similarity_threshold and max_similarity > best_score:
            best_score = max_similarity
            best_match = existing_poi

    if best_match:
        print(f"      [FUZZY] '{poi_name}' matched with '{best_match.poi_name}' ({best_score:.1f}%)")
    
    
    return (best_match)

# increments count and updates most recent date.
def merge_poi_counts(existing_poi: ExtractedPOI, new_post: Post, 
                     new_activity: str = None) -> None:
    existing_poi.count += 1
    
    # append activity if provided and different
    if new_activity and new_activity.strip():
        if existing_poi.poi_activity:
            if new_activity.lower() not in existing_poi.poi_activity.lower():
                existing_poi.poi_activity += f"; {new_activity}"
        else:
            existing_poi.poi_activity = new_activity
    
    existing_post = db.session.get(Post, existing_poi.post_id)
    if existing_post and new_post.date and existing_post.date:
        if new_post.date > existing_post.date:
            existing_poi.post_id = new_post.post_id
    
    db.session.commit()