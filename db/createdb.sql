CREATE TABLE account (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    followers INT
);

CREATE TABLE post (
    post_id SERIAL PRIMARY KEY,
    account_id INT REFERENCES account(id) ON DELETE CASCADE,
    date TIMESTAMP,
    likes INT,
    ---comment_count INT,
    caption TEXT,
    poi_extracted BOOLEAN
);

CREATE TABLE location (
    post_id INT PRIMARY KEY REFERENCES post(post_id) ON DELETE CASCADE,
    loc_url TEXT,
    loc_name TEXT,
    latitude FLOAT,
    longitude FLOAT,
    city TEXT
);

CREATE TABLE extractedpois (
    id SERIAL PRIMARY KEY,
    post_id INT REFERENCES post(post_id) ON DELETE CASCADE,
    poi_name TEXT,
    poi_type TEXT,
    poi_activity TEXT,
    count INT,
    ---score FLOAT
);

--for faster join
--CREATE INDEX idx_poi_name ON extractedpois (lower(poi_name));
--CREATE INDEX idx_extracted_post ON extractedpois (post_id);
