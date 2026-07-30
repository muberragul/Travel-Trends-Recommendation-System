"""
Performance Benchmarking Script for Travel Trends System
Tests extraction time (LLM vs NER) and query response time
"""

import os
import time
import json
import statistics
from pathlib import Path
from typing import List, Dict
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from models import db, Post, Account, Location, ExtractedPOI
from extract_poi import extract_pois_with_qwen, extract_with_spacy, QWEN_KEY
from load_data import DataLoader
from sqlalchemy import func, desc
import math

class PerformanceBenchmark:
    def __init__(self):
        self.benchmark_output_dir = PROJECT_ROOT / 'results' / 'benchmarks'
        self.results = {
            'llm_extraction': [],
            'ner_extraction': [],
            'query_response': []
        }
    
    def clear_test_data(self):
        """Clear existing test data from database"""
        with app.app_context():
            # Delete in correct order due to foreign keys
            ExtractedPOI.query.filter(
                ExtractedPOI.post_id.in_(
                    db.session.query(Post.post_id).filter(Post.caption.like('%Courtauld Gallery%'))
                )
            ).delete(synchronize_session=False)
            
            Post.query.filter(Post.caption.like('%Courtauld Gallery%')).delete(synchronize_session=False)
            db.session.commit()
            print("[INFO] Test data cleared from database")
    
    def load_test_data(self):
        """Load test CSV data"""
        with app.app_context():
            loader = DataLoader(str(PROJECT_ROOT / 'data'))
            # Load only the specific file by using its exact name as pattern
            loader.load_all_csvs(pattern='synthetic_instagram_travel_data_london_more_part1.csv')
            
            posts_loaded = loader.stats['posts_added']
            print(f"[INFO] Loaded {posts_loaded} posts for testing")
            
            if posts_loaded == 0:
                print(f"[WARNING] No new posts loaded. {loader.stats['posts_skipped']} duplicates skipped")
            
            return posts_loaded
    
    def benchmark_llm_extraction(self, iterations: int = 3):
        """Benchmark LLM-based POI extraction"""
        print(f"\n{'='*60}")
        print("BENCHMARKING LLM EXTRACTION (Qwen)")
        print(f"{'='*60}")
        
        if not QWEN_KEY:
            print("[WARNING] No QWEN_API_KEY found, skipping LLM benchmark")
            return
        
        with app.app_context():
            posts = Post.query.filter_by(poi_extracted=False).limit(100).all()
            
            for iteration in range(1, iterations + 1):
                print(f"\n[RUN {iteration}/{iterations}] Processing {len(posts)} posts...")
                
                start_time = time.time()
                total_pois = 0
                successful_extractions = 0
                
                for i, post in enumerate(posts):
                    caption = post.caption or ""
                    loc_name = post.location.loc_name if post.location else ""
                    
                    try:
                        pois = extract_pois_with_qwen(caption, location=loc_name)
                        total_pois += len(pois)
                        if pois:
                            successful_extractions += 1
                    except Exception as e:
                        print(f"[ERROR] Post {i+1}: {e}")
                    
                    if (i + 1) % 25 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Progress: {i+1}/100 posts | Elapsed: {elapsed:.2f}s")
                
                end_time = time.time()
                total_time = end_time - start_time
                
                self.results['llm_extraction'].append({
                    'iteration': iteration,
                    'total_time': total_time,
                    'posts_processed': len(posts),
                    'pois_extracted': total_pois,
                    'successful_posts': successful_extractions,
                    'avg_time_per_post': total_time / len(posts),
                    'pois_per_post': total_pois / len(posts)
                })
                
                print(f"\n[RESULTS] Run {iteration}:")
                print(f"  Total Time: {total_time:.2f}s")
                print(f"  Avg Time/Post: {total_time/len(posts):.3f}s")
                print(f"  POIs Extracted: {total_pois}")
                print(f"  Success Rate: {successful_extractions/len(posts)*100:.1f}%")
    
    def benchmark_ner_extraction(self, iterations: int = 3):
        """Benchmark spaCy NER-based extraction"""
        print(f"\n{'='*60}")
        print("BENCHMARKING NER EXTRACTION (spaCy)")
        print(f"{'='*60}")
        
        with app.app_context():
            posts = Post.query.limit(100).all()
            
            for iteration in range(1, iterations + 1):
                print(f"\n[RUN {iteration}/{iterations}] Processing {len(posts)} posts...")
                
                start_time = time.time()
                total_pois = 0
                successful_extractions = 0
                
                for i, post in enumerate(posts):
                    caption = post.caption or ""
                    loc_name = post.location.loc_name if post.location else ""
                    
                    try:
                        pois = extract_with_spacy(caption, loc_name)
                        total_pois += len(pois)
                        if pois:
                            successful_extractions += 1
                    except Exception as e:
                        print(f"[ERROR] Post {i+1}: {e}")
                    
                    if (i + 1) % 25 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Progress: {i+1}/100 posts | Elapsed: {elapsed:.2f}s")
                
                end_time = time.time()
                total_time = end_time - start_time
                
                self.results['ner_extraction'].append({
                    'iteration': iteration,
                    'total_time': total_time,
                    'posts_processed': len(posts),
                    'pois_extracted': total_pois,
                    'successful_posts': successful_extractions,
                    'avg_time_per_post': total_time / len(posts),
                    'pois_per_post': total_pois / len(posts)
                })
                
                print(f"\n[RESULTS] Run {iteration}:")
                print(f"  Total Time: {total_time:.2f}s")
                print(f"  Avg Time/Post: {total_time/len(posts):.3f}s")
                print(f"  POIs Extracted: {total_pois}")
                print(f"  Success Rate: {successful_extractions/len(posts)*100:.1f}%")

    def benchmark_query_performance(self, iterations: int = 3):
        """Benchmark query response time for first 50 results"""
        print(f"\n{'='*60}")
        print("BENCHMARKING QUERY RESPONSE TIME")
        print(f"{'='*60}")
        
        with app.app_context():
            # First, populate ExtractedPOI table if empty
            poi_count = ExtractedPOI.query.count()
            if poi_count == 0:
                print("[INFO] No POIs in database, running extraction first...")
                posts = Post.query.filter_by(poi_extracted=False).limit(100).all()
                for post in posts:
                    caption = post.caption or ""
                    loc_name = post.location.loc_name if post.location else ""
                    
                    if QWEN_KEY:
                        pois = extract_pois_with_qwen(caption, location=loc_name)
                    else:
                        pois = extract_with_spacy(caption, loc_name)
                    
                    account = post.account
                    followers = account.followers if account else 0
                    
                    for r in pois:
                        poi_name = (r.get("poi_name") or "").strip()
                        if not poi_name:
                            continue
                        
                        exists = ExtractedPOI.query.filter_by(
                            post_id=post.post_id, 
                            poi_name=poi_name
                        ).first()
                        if not exists:
                            score = (post.likes or 0) / max((followers or 1), 1) * 100.0
                            new_poi = ExtractedPOI(
                                post_id=post.post_id,
                                poi_name=poi_name,
                                poi_type=r.get("poi_type", "other"),
                                poi_activity=r.get("activity", ""),
                                count=1,
                                score=score
                            )
                            db.session.add(new_poi)
                    
                    post.poi_extracted = True
                db.session.commit()
                print(f"[INFO] Populated {ExtractedPOI.query.count()} POIs")
            
            # Benchmark query performance using exact app.py logic
            from datetime import datetime, timedelta
            
            for iteration in range(1, iterations + 1):
                print(f"\n[RUN {iteration}/{iterations}] Querying top 50 POIs...")
                
                start_time = time.time()
                
                # Parameters (same as app.py defaults)
                city = "London"
                limit = 50
                days = 365
                date_threshold = datetime.now() - timedelta(days=days)
                
                # EXACT QUERY FROM app.py
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
                    .filter(Post.date >= date_threshold)
                    .group_by(ExtractedPOI.poi_name, ExtractedPOI.poi_type, ExtractedPOI.poi_activity)
                    .all()
                )
                
                # EXACT IMPACT SCORE CALCULATION FROM app.py
                def impact_score(mentions, total_likes, avg_followers, most_recent_date):
                    avg_followers = float(avg_followers or 1)
                    total_likes = float(total_likes or 0)
                    mentions = int(mentions or 0)
                    
                    engagement_rate = total_likes / avg_followers
                    mention_factor = 1.0 + math.log1p(mentions)
                    base_score = engagement_rate * mention_factor

                    if most_recent_date:
                        if isinstance(most_recent_date, datetime):
                            most_recent_date = most_recent_date.date()    
                        days_ago = (datetime.now().date() - most_recent_date).days
                        
                        if days_ago <= 30:
                            recency_multiplier = 1.5
                        elif days_ago <= 90:
                            recency_multiplier = 1.3
                        elif days_ago <= 180:
                            recency_multiplier = 1.1
                        else:
                            recency_multiplier = 1.0
                    else:
                        recency_multiplier = 1.0    
                    return base_score * recency_multiplier
                
                results = []
                for r in rows:
                    sc = impact_score(r.mentions, r.total_likes, r.avg_followers, r.most_recent_date)
                    results.append({
                        "poi_name": r.poi_name,
                        "poi_type": r.poi_type or "other",
                        "poi_activity": r.poi_activity or "N/A",
                        "mentions": int(r.mentions),
                        "total_likes": int(r.total_likes or 0),
                        "avg_likes": float(r.avg_likes or 0.0),
                        "avg_followers": float(r.avg_followers or 0.0),
                        "impact": round(sc, 6),
                    })
                
                # Sort and paginate (same as app.py)
                results_sorted = sorted(results, key=lambda x: x["impact"], reverse=True)
                top_50 = results_sorted[:limit]
                
                end_time = time.time()
                total_time = end_time - start_time
                
                avg_impact = statistics.mean([x['impact'] for x in top_50]) if top_50 else 0
                
                self.results['query_response'].append({
                    'iteration': iteration,
                    'query_time': total_time,
                    'total_pois': len(results_sorted),
                    'returned_pois': len(top_50),
                    'avg_impact_score': avg_impact
                })
                
                print(f"\n[RESULTS] Run {iteration}:")
                print(f"  Query Time: {total_time*1000:.2f}ms")
                print(f"  Total POIs Found: {len(results_sorted)}")
                print(f"  Returned: {len(top_50)}")
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print(f"\n{'='*80}")
        print("PERFORMANCE BENCHMARK REPORT")
        print(f"{'='*80}")
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Dataset: 100 Instagram posts from London")
        print(f"Iterations: 3 runs per method")
        
        # LLM Extraction Results
        if self.results['llm_extraction']:
            print(f"\n{'='*80}")
            print("1. LLM-BASED EXTRACTION (Qwen Turbo)")
            print(f"{'='*80}")
            print(f"\n{'Run':<6} {'Total Time':<12} {'Avg/Post':<12} {'POIs':<8} {'POIs/Post':<12} {'Success Rate':<15}")
            print("-" * 80)
            
            for r in self.results['llm_extraction']:
                print(f"{r['iteration']:<6} "
                      f"{r['total_time']:>10.2f}s  "
                      f"{r['avg_time_per_post']*1000:>9.1f}ms  "
                      f"{r['pois_extracted']:<8} "
                      f"{r['pois_per_post']:>10.2f}  "
                      f"{r['successful_posts']/r['posts_processed']*100:>13.1f}%")
            
            # Statistics
            times = [r['total_time'] for r in self.results['llm_extraction']]
            avg_per_post = [r['avg_time_per_post']*1000 for r in self.results['llm_extraction']]
            
            print(f"\nStatistics:")
            print(f"  Mean Total Time: {statistics.mean(times):.2f}s ± {statistics.stdev(times):.2f}s")
            print(f"  Mean Time/Post: {statistics.mean(avg_per_post):.1f}ms ± {statistics.stdev(avg_per_post):.1f}ms")
        
        # NER Extraction Results
        if self.results['ner_extraction']:
            print(f"\n{'='*80}")
            print("2. NER-BASED EXTRACTION (spaCy)")
            print(f"{'='*80}")
            print(f"\n{'Run':<6} {'Total Time':<12} {'Avg/Post':<12} {'POIs':<8} {'POIs/Post':<12} {'Success Rate':<15}")
            print("-" * 80)
            
            for r in self.results['ner_extraction']:
                print(f"{r['iteration']:<6} "
                      f"{r['total_time']:>10.2f}s  "
                      f"{r['avg_time_per_post']*1000:>9.1f}ms  "
                      f"{r['pois_extracted']:<8} "
                      f"{r['pois_per_post']:>10.2f}  "
                      f"{r['successful_posts']/r['posts_processed']*100:>13.1f}%")
            
            # Statistics
            times = [r['total_time'] for r in self.results['ner_extraction']]
            avg_per_post = [r['avg_time_per_post']*1000 for r in self.results['ner_extraction']]
            
            print(f"\nStatistics:")
            print(f"  Mean Total Time: {statistics.mean(times):.2f}s ± {statistics.stdev(times):.2f}s")
            print(f"  Mean Time/Post: {statistics.mean(avg_per_post):.1f}ms ± {statistics.stdev(avg_per_post):.1f}ms")
        
        # Comparison
        if self.results['llm_extraction'] and self.results['ner_extraction']:
            print(f"\n{'='*80}")
            print("3. COMPARATIVE ANALYSIS")
            print(f"{'='*80}")
            
            llm_mean = statistics.mean([r['avg_time_per_post']*1000 for r in self.results['llm_extraction']])
            ner_mean = statistics.mean([r['avg_time_per_post']*1000 for r in self.results['ner_extraction']])
            
            llm_pois = statistics.mean([r['pois_per_post'] for r in self.results['llm_extraction']])
            ner_pois = statistics.mean([r['pois_per_post'] for r in self.results['ner_extraction']])
            
            speedup = llm_mean / ner_mean
            
            print(f"\n{'Metric':<20} {'LLM (Qwen)':<15} {'NER (spaCy)':<15} {'Difference':<20}")
            print("-" * 80)
            print(f"{'Avg Time/Post':<20} {llm_mean:>13.1f}ms {ner_mean:>14.1f}ms  {speedup:.2f}x {'slower' if speedup > 1 else 'faster'}")
            print(f"{'Avg POIs/Post':<20} {llm_pois:>13.2f}   {ner_pois:>14.2f}    {((llm_pois/ner_pois-1)*100):+.1f}%")
        
        # Query Performance
        if self.results['query_response']:
            print(f"\n{'='*80}")
            print("4. QUERY RESPONSE TIME (Top 50 Results)")
            print(f"{'='*80}")
            print(f"\n{'Run':<6} {'Query Time':<15} {'Total POIs':<12} {'Returned':<10} {'Avg Impact Score':<18}")
            print("-" * 80)
            
            for r in self.results['query_response']:
                print(f"{r['iteration']:<6} "
                      f"{r['query_time']*1000:>12.2f}ms  "
                      f"{r['total_pois']:<12} "
                      f"{r['returned_pois']:<10} "
                      f"{r['avg_impact_score']:>16.4f}")
            
            # Statistics
            times = [r['query_time']*1000 for r in self.results['query_response']]
            
            print(f"\nStatistics:")
            print(f"  Mean Query Time: {statistics.mean(times):.2f}ms ± {statistics.stdev(times):.2f}ms")
            print(f"  Min Query Time: {min(times):.2f}ms")
            print(f"  Max Query Time: {max(times):.2f}ms")
        
        print(f"\n{'='*80}")
        print("END OF REPORT")
        print(f"{'='*80}\n")
        
        # Save to file
        self.save_results_to_file()
    
    def save_results_to_file(self):
        """Save results to JSON file for further analysis"""
        self.benchmark_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.benchmark_output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"[INFO] Results saved to {output_file}")


def main():
    """Run complete benchmark suite"""
    benchmark = PerformanceBenchmark()
    
    print("="*80)
    print("TRAVEL TRENDS SYSTEM - PERFORMANCE BENCHMARK")
    print("="*80)
    print("\nThis benchmark will:")
    print("  1. Clear existing test data")
    print("  2. Load 100 test posts from CSV")
    print("  3. Run LLM extraction (3 iterations)")
    print("  4. Run NER extraction (3 iterations)")
    print("  5. Benchmark query performance (3 iterations)")
    print("  6. Generate comprehensive report\n")
    
    input("Press Enter to continue...")
    
    # Step 1: Clear and load data
    benchmark.clear_test_data()
    posts_loaded = benchmark.load_test_data()
    
    if posts_loaded == 0:
        print("[ERROR] No posts loaded. Exiting.")
        return
    
    # Step 2: Run benchmarks
    benchmark.benchmark_llm_extraction(iterations=3)
    
    # Clear POI flags for NER test
    with app.app_context():
        Post.query.update({Post.poi_extracted: False})
        db.session.commit()
    
    benchmark.benchmark_ner_extraction(iterations=3)
    
    # Step 3: Query performance
    benchmark.benchmark_query_performance(iterations=3)
    
    # Step 4: Generate report
    benchmark.generate_report()


if __name__ == "__main__":
    main()