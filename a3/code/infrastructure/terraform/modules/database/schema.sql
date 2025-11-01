-- ArXplorer Database Schema
-- This file defines the database structure for the academic search assistant

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS arxplorer CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE arxplorer;

-- Papers table - stores original paper metadata
CREATE TABLE IF NOT EXISTS papers (
    id VARCHAR(50) PRIMARY KEY,  -- ArXiv ID
    title TEXT NOT NULL,
    authors TEXT,
    abstract TEXT,
    categories VARCHAR(255),
    doi VARCHAR(100),
    journal_ref TEXT,
    arxiv_url VARCHAR(255),
    pdf_url VARCHAR(255),
    submitted_date DATE,
    updated_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_categories (categories),
    INDEX idx_submitted_date (submitted_date),
    FULLTEXT idx_title_abstract (title, abstract)
) ENGINE=InnoDB;

-- Processed papers table - stores cleaned and processed text
CREATE TABLE IF NOT EXISTS processed_papers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    paper_id VARCHAR(50) NOT NULL,
    cleaned_title TEXT,
    cleaned_abstract TEXT,
    keywords TEXT,
    summary TEXT,
    word_count INT,
    processing_version VARCHAR(20),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    INDEX idx_paper_id (paper_id),
    INDEX idx_processing_version (processing_version),
    FULLTEXT idx_cleaned_content (cleaned_title, cleaned_abstract, keywords)
) ENGINE=InnoDB;

-- Embeddings table - stores vector embeddings metadata
CREATE TABLE IF NOT EXISTS embeddings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    paper_id VARCHAR(50) NOT NULL,
    embedding_type VARCHAR(50) NOT NULL,  -- 'scibert', 'title', 'abstract'
    vector_dimension INT NOT NULL,
    s3_path VARCHAR(500),  -- Path to actual embedding file in S3
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    INDEX idx_paper_id (paper_id),
    INDEX idx_embedding_type (embedding_type),
    INDEX idx_model_version (model_version),
    UNIQUE KEY unique_paper_embedding (paper_id, embedding_type, model_version)
) ENGINE=InnoDB;

-- Search results cache - stores frequently accessed search results
CREATE TABLE IF NOT EXISTS search_cache (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_hash VARCHAR(64) NOT NULL,  -- MD5 hash of search query
    query_text TEXT NOT NULL,
    results JSON,  -- JSON array of paper IDs and scores
    result_count INT,
    search_type VARCHAR(50),  -- 'semantic', 'keyword', 'hybrid'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    
    INDEX idx_query_hash (query_hash),
    INDEX idx_search_type (search_type),
    INDEX idx_created_at (created_at),
    INDEX idx_expires_at (expires_at)
) ENGINE=InnoDB;

-- User queries log - for analytics and improvement
CREATE TABLE IF NOT EXISTS user_queries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100),
    query_text TEXT NOT NULL,
    search_type VARCHAR(50),
    result_count INT,
    response_time_ms INT,
    user_ip VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_session_id (session_id),
    INDEX idx_search_type (search_type),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

-- Processing jobs table - track pipeline processing status
CREATE TABLE IF NOT EXISTS processing_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL,  -- 'ingestion', 'embedding', 'indexing'
    status VARCHAR(20) NOT NULL,    -- 'pending', 'running', 'completed', 'failed'
    input_path VARCHAR(500),
    output_path VARCHAR(500),
    parameters JSON,
    progress_percent INT DEFAULT 0,
    records_processed INT DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_job_type (job_type),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

-- System configuration table
CREATE TABLE IF NOT EXISTS system_config (
    config_key VARCHAR(100) PRIMARY KEY,
    config_value TEXT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Insert default configuration values
INSERT INTO system_config (config_key, config_value, description) VALUES
('embedding_model', 'allenai/scibert_scivocab_uncased', 'Current embedding model being used'),
('vector_dimension', '768', 'Dimension of embedding vectors'),
('index_version', '1.0', 'Current FAISS index version'),
('last_full_reindex', NULL, 'Timestamp of last complete reindexing'),
('max_search_results', '100', 'Maximum number of search results to return')
ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP;

-- Create views for common queries
CREATE OR REPLACE VIEW recent_papers AS
SELECT p.*, pp.word_count, pp.processed_at
FROM papers p
LEFT JOIN processed_papers pp ON p.id = pp.paper_id
WHERE p.submitted_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
ORDER BY p.submitted_date DESC;

CREATE OR REPLACE VIEW embedding_stats AS
SELECT 
    embedding_type,
    model_version,
    COUNT(*) as count,
    MIN(created_at) as first_created,
    MAX(created_at) as last_created
FROM embeddings
GROUP BY embedding_type, model_version;

-- Stored procedures for common operations
DELIMITER //

CREATE PROCEDURE GetPaperWithEmbeddings(IN paper_id VARCHAR(50))
BEGIN
    SELECT p.*, pp.cleaned_title, pp.cleaned_abstract, pp.keywords,
           e.embedding_type, e.s3_path, e.model_version
    FROM papers p
    LEFT JOIN processed_papers pp ON p.id = pp.paper_id
    LEFT JOIN embeddings e ON p.id = e.paper_id
    WHERE p.id = paper_id;
END //

CREATE PROCEDURE UpdateProcessingJob(
    IN job_id INT,
    IN new_status VARCHAR(20),
    IN progress INT,
    IN records INT,
    IN error_msg TEXT
)
BEGIN
    UPDATE processing_jobs 
    SET status = new_status,
        progress_percent = progress,
        records_processed = records,
        error_message = error_msg,
        completed_at = CASE WHEN new_status IN ('completed', 'failed') THEN NOW() ELSE completed_at END
    WHERE id = job_id;
END //

DELIMITER ;

-- Create indexes for better performance
ALTER TABLE papers ADD INDEX idx_title_length ((CHAR_LENGTH(title)));
ALTER TABLE processed_papers ADD INDEX idx_word_count (word_count);
ALTER TABLE user_queries ADD INDEX idx_response_time (response_time_ms);

-- Grant permissions (will be handled by application)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON arxplorer.* TO 'app_user'@'%';

COMMIT;