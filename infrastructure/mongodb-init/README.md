# MongoDB Initialization

## 🗂️ **Files in this directory:**

### `init-arxplorer.js`
- **Purpose**: MongoDB initialization script for local development
- **When used**: Only when running MongoDB locally with Docker
- **What it does**: 
  - Creates ArXplorer database
  - Creates user with read/write permissions
  - Creates all required collections
  - Sets up performance indexes

## 🚨 **Important Notes:**

### **For MongoDB Atlas (Cloud) Users:**
- ✅ **You don't need this script** - you're using MongoDB Atlas
- ✅ Your collections are created automatically by the Python pipeline
- ✅ Indexes are created by `mongodb_integration.py`

### **For Local Development Users:**
- 🐳 Use `docker-compose.mongodb.yml` to run local MongoDB
- 📜 This script runs automatically on first container startup
- 🔧 Creates proper database structure and indexes

## 🔧 **Usage:**

### Local MongoDB with Docker:
```bash
# Start local MongoDB with initialization
docker-compose -f docker-compose.mongodb.yml up -d

# The init script runs automatically on first startup
```

### Manual Execution:
```bash
# Connect to MongoDB shell
mongosh mongodb://localhost:27017

# Execute the script
load('./mongodb-init/init-arxplorer.js')
```

## 📊 **What Gets Created:**

### Collections:
- `raw_papers` - Original ArXiv papers
- `processed_papers` - Cleaned and analyzed papers
- `embeddings` - Vector embeddings for search
- `search_queries` - Search history
- `pipeline_runs` - Pipeline execution logs
- `user_feedback` - User ratings and feedback

### Indexes:
- **Performance indexes** on commonly queried fields
- **Unique indexes** on ID fields to prevent duplicates
- **Compound indexes** for complex queries

## 🔄 **Current Status:**

**Your Setup**: MongoDB Atlas (Cloud)
**This Script**: Not actively used (but ready for local development)
**Alternative**: Use `enhanced_pipeline.py` - it creates collections automatically