"""   
python -c "from load_data import load_from_folder; load_from_folder('data')"

Usage:
    load_from_folder("data")
    load_from_folder("data", "munich*.csv")
    load_from_folder("data/cities", "*.csv")
"""

import pandas as pd
from pathlib import Path
from app import app
from models import db, Account, Post, Location

CITY_BOUNDARIES = {
    'London': [51.2868, 51.6919, -0.5103, 0.3340],  
    'Munich': [48.061, 48.248, 11.360, 11.723],
    'Barcelona': [41.320, 41.469, 2.052, 2.228],
    'Florence': [43.727, 43.832, 11.154, 11.328],
    'Istanbul': [40.978, 41.108, 28.872, 29.154],
    'Dublin': [53.245, 53.385, -6.450, -6.150],
    'Sydney': [-34.118, -33.578, 150.520, 151.343],
}

class DataLoader:
    def __init__(self, data_folder: str = "data"):
        self.data_folder = Path(data_folder)
        self.stats = {
            'files_processed': 0,
            'posts_added': 0,
            'posts_skipped': 0,
            'accounts_added': 0,
            'errors': []
        }
    
    def load_all_csvs(self, pattern: str = "*.csv", batch_size: int = 500):
        if not self.data_folder.exists():
            print(f"[ERROR] Data folder not found: {self.data_folder}")
            return
        
        all_files = []
        for ext in ['*.csv', '*.txt']:
            all_files.extend(self.data_folder.glob(ext))
        
        if not all_files:
            print(f"[WARNING] No CSV/TXT files found in {self.data_folder}")
            return
        
        print(f"[INFO] Found {len(all_files)} CSV files to process")
        
        with app.app_context():
            for file_path in all_files:
                print(f"\n[PROCESSING] {file_path.name}")
                try:
                    self._load_single_csv(file_path, batch_size)
                except Exception as e:
                    error_msg = f"Failed to load {file_path.name}: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    self.stats['errors'].append(error_msg)
            
            self._print_summary()
    
    def _load_single_csv(self, file_path: Path, batch_size: int):
        try:
            df = pd.read_csv(
                file_path, 
                encoding='utf-8',
                on_bad_lines='skip',  
                engine='python',       
                quotechar='"',
                escapechar='\\'
            )
            
            if len(df.columns) <= 1:
                raise ValueError("Failed to parse with comma separator")
                
        except Exception as e:
            print(f"[WARNING] CSV parsing failed, trying alternative separators: {e}")
            for sep in ['\t', ';', '|']:
                try:
                    df = pd.read_csv(
                        file_path, 
                        sep=sep, 
                        encoding='utf-8',
                        on_bad_lines='skip',
                        engine='python'
                    )
                    if len(df.columns) > 1:
                        print(f"[INFO] Successfully parsed with separator: {repr(sep)}")
                        break
                except:
                    continue
            else:
                raise ValueError(f"Could not parse file")
    
        # processing data
        required_cols = ['username', 'followers', 'caption', 'date', 'likes', 
                        'latitude', 'longitude', 'location_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # clean data
        df = df.dropna(subset=['username', 'caption', 'latitude', 'longitude'])
        
        for start_idx in range(0, len(df), batch_size):
            batch = df.iloc[start_idx:start_idx + batch_size]
            self._process_batch(batch)
        
        self.stats['files_processed'] += 1
    
    # process and commit to database.
    def _process_batch(self, batch: pd.DataFrame):
        for _, row in batch.iterrows():
            try:
                account = Account.query.filter_by(username=row['username']).first()
                if not account:
                    account = Account(
                        username=row['username'],
                        followers=int(row['followers'])
                    )
                    db.session.add(account)
                    db.session.flush()
                    self.stats['accounts_added'] += 1
                
                # check if post already exists
                post_date = pd.to_datetime(row['date'])
                existing_post = Post.query.filter_by(
                    account_id=account.id,
                    caption=row['caption'],
                    date=post_date
                ).first()
                
                if existing_post:
                    self.stats['posts_skipped'] += 1
                    continue
                
                post = Post(
                    account_id=account.id,
                    caption=str(row['caption']),
                    date=post_date,
                    likes=int(row['likes']) if pd.notna(row['likes']) else 0,
                    poi_extracted=False
                )
                db.session.add(post)
                db.session.flush()

                # get city from coordinates from the given bounding boxes
                city = self.get_city_from_coord(
                    float(row['latitude']), 
                    float(row['longitude'])
                )

                location = Location(
                    post_id=post.post_id,
                    loc_name=str(row['location_name']),
                    loc_url=str(row.get('location_url', '')),
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude']),
                    city=city
                )
                db.session.add(location)
                
                self.stats['posts_added'] += 1
                
            except Exception as e:
                error_msg = f"Row error: {str(e)}"
                print(f"[WARNING] {error_msg}")
                self.stats['errors'].append(error_msg)
                db.session.rollback()
                continue
        db.session.commit()

    # get city name from latitude/longitude coordinates using bounding boxes
    @staticmethod
    def get_city_from_coord(lat: float, lon: float) -> str:
        for city, (lat_min, lat_max, lon_min, lon_max) in CITY_BOUNDARIES.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return city
        return "Unknown"
    
    def _print_summary(self):
        print("\n" + "="*60)
        print("DATA LOADING SUMMARY")
        print("="*60)
        print(f"Files processed:    {self.stats['files_processed']}")
        print(f"New accounts:       {self.stats['accounts_added']}")
        print(f"Posts added:        {self.stats['posts_added']}")
        print(f"Posts skipped:      {self.stats['posts_skipped']} (duplicates)")
        print(f"Errors:             {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            print("\nError details (first 5):")
            for err in self.stats['errors'][:5]:
                print(f"  - {err}")
        print("="*60 + "\n")


def load_from_folder(folder_path: str = "data", pattern: str = "*.csv"):
    loader = DataLoader(folder_path)
    loader.load_all_csvs(pattern=pattern)


if __name__ == "__main__":
    load_from_folder("data")
