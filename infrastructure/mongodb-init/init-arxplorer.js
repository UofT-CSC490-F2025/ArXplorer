// MongoDB initialization script for ArXplorer
// This script runs when the MongoDB container starts for the first time

print("Initializing ArXplorer MongoDB database...")

// Switch to arxplorer database
db = db.getSiblingDB('arxplorer')

// Create a user for the arxplorer database
db.createUser({
  user: "arxplorer_user",
  pwd: process.env.MONGODB_PASSWORD || "arxplorer_secure_password_2024",
  roles: [
    {
      role: "readWrite",
      db: "arxplorer"
    }
  ]
})

// Create collections and indexes (without dummy data)
print("Creating collections and indexes...")

// Create collections explicitly
db.createCollection("raw_papers")
db.createCollection("processed_papers") 
db.createCollection("embeddings")
db.createCollection("search_queries")
db.createCollection("pipeline_runs")
db.createCollection("user_feedback")

// Create indexes for better performance
print("Creating indexes...")

// Raw papers indexes
db.raw_papers.createIndex({ "arxiv_id": 1 }, { unique: true })
db.raw_papers.createIndex({ "categories": 1 })
db.raw_papers.createIndex({ "submitted_date": -1 })
db.raw_papers.createIndex({ "status": 1 })
db.raw_papers.createIndex({ "batch_id": 1 })

// Processed papers indexes  
db.processed_papers.createIndex({ "arxiv_id": 1 }, { unique: true })
db.processed_papers.createIndex({ "word_count": 1 })
db.processed_papers.createIndex({ "language": 1 })
db.processed_papers.createIndex({ "batch_id": 1 })

// Embeddings indexes
db.embeddings.createIndex({ "arxiv_id": 1 }, { unique: true })
db.embeddings.createIndex({ "model_name": 1 })
db.embeddings.createIndex({ "batch_id": 1 })
db.embeddings.createIndex({ "created_at": -1 })

// Search queries indexes
db.search_queries.createIndex({ "query_id": 1 }, { unique: true })
db.search_queries.createIndex({ "created_at": -1 })

// Pipeline runs indexes
db.pipeline_runs.createIndex({ "batch_id": 1 }, { unique: true })
db.pipeline_runs.createIndex({ "start_time": -1 })
db.pipeline_runs.createIndex({ "status": 1 })

// User feedback indexes
db.user_feedback.createIndex({ "feedback_id": 1 }, { unique: true })
db.user_feedback.createIndex({ "created_at": -1 })
db.user_feedback.createIndex({ "rating": 1 })

print("✅ Collections created successfully!")
print("✅ Indexes created successfully!")
print("✅ ArXplorer MongoDB database initialized successfully!")

// Show database stats
print("\n📊 Database Statistics:")
print("Collections:", db.getCollectionNames().length)
printjson(db.stats())