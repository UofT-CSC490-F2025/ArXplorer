#!/usr/bin/env python3
"""
MongoDB Data Viewer - Browse and query your ArXplorer database
"""

import asyncio
import logging
import yaml
from datetime import datetime
from mongodb_integration import MongoDBManager, MongoDBConfig
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Could not load config file: {e}")
        return {}

async def browse_collections():
    """Browse all collections and show sample data"""
    try:
        # Load configuration
        config_data = load_config()
        mongodb_config = config_data.get('mongodb', {})
        # Only use the core MongoDBConfig fields
        config = MongoDBConfig(
            connection_string=mongodb_config.get('connection_string', 'mongodb://localhost:27017/'),
            database_name=mongodb_config.get('database_name', 'arxplorer'),
            use_async=mongodb_config.get('use_async', True)
        )
        
        # Initialize MongoDB manager
        mongo_manager = MongoDBManager(config)
        await mongo_manager.initialize_database()
        
        print("=" * 80)
        print("🔍 ARXPLORER MONGODB DATABASE BROWSER")
        print("=" * 80)
        
        # Check each collection
        collections = ['raw_papers', 'processed_papers', 'embeddings', 'search_queries', 'pipeline_runs']
        
        for collection_name in collections:
            print(f"\n📊 COLLECTION: {collection_name.upper()}")
            print("-" * 50)
            
            if config.use_async:
                collection = mongo_manager.async_db[collection_name]
                count = await collection.count_documents({})
                print(f"Total documents: {count}")
                
                if count > 0:
                    # Show first document
                    sample = await collection.find_one({})
                    if sample:
                        # Remove _id for cleaner display
                        if '_id' in sample:
                            del sample['_id']
                        
                        print("\n📄 Sample Document:")
                        # Pretty print the document
                        for key, value in sample.items():
                            if key in ['title_embedding', 'abstract_embedding', 'combined_embedding']:
                                print(f"  {key}: [Vector with {len(value)} dimensions]")
                            elif isinstance(value, list) and len(value) > 3:
                                print(f"  {key}: [{value[0]}, {value[1]}, {value[2]}, ... ({len(value)} items)]")
                            elif isinstance(value, str) and len(value) > 100:
                                print(f"  {key}: {value[:100]}...")
                            else:
                                print(f"  {key}: {value}")
                    
                    # Show recent documents
                    if collection_name in ['raw_papers', 'processed_papers']:
                        print(f"\n📋 Recent {collection_name.replace('_', ' ').title()}:")
                        cursor = collection.find({}).sort("inserted_at", -1).limit(3)
                        async for doc in cursor:
                            if '_id' in doc:
                                del doc['_id']
                            if collection_name == 'raw_papers':
                                print(f"  • {doc.get('arxiv_id', 'N/A')}: {doc.get('title', 'No title')[:50]}...")
                            elif collection_name == 'processed_papers':
                                print(f"  • {doc.get('arxiv_id', 'N/A')}: {doc.get('word_count', 'N/A')} words, {doc.get('language', 'N/A')}")
                else:
                    print("  (Empty collection)")
        
        # Database statistics
        print(f"\n📈 DATABASE STATISTICS")
        print("-" * 50)
        if config.use_async:
            db_stats = await mongo_manager.async_db.command("dbStats")
            print(f"Database: {config.database_name}")
            print(f"Collections: {db_stats.get('collections', 'N/A')}")
            print(f"Data Size: {db_stats.get('dataSize', 0) / 1024:.2f} KB")
            print(f"Index Size: {db_stats.get('indexSize', 0) / 1024:.2f} KB")
        
        # Close connections
        mongo_manager.close_connections()
        print(f"\n✅ Database browsing completed")
        
    except Exception as e:
        logger.error(f"❌ Failed to browse database: {e}")
        return False
    
    return True

async def query_papers(search_term: str = None):
    """Query papers with optional search term"""
    try:
        # Load configuration
        config_data = load_config()
        mongodb_config = config_data.get('mongodb', {})
        # Only use the core MongoDBConfig fields
        config = MongoDBConfig(
            connection_string=mongodb_config.get('connection_string', 'mongodb://localhost:27017/'),
            database_name=mongodb_config.get('database_name', 'arxplorer'),
            use_async=mongodb_config.get('use_async', True)
        )
        
        # Initialize MongoDB manager
        mongo_manager = MongoDBManager(config)
        await mongo_manager.initialize_database()
        
        print(f"\n🔍 SEARCHING PAPERS" + (f" (term: '{search_term}')" if search_term else ""))
        print("-" * 50)
        
        if config.use_async:
            collection = mongo_manager.async_db['raw_papers']
            
            # Build query
            if search_term:
                query = {
                    "$or": [
                        {"title": {"$regex": search_term, "$options": "i"}},
                        {"abstract": {"$regex": search_term, "$options": "i"}},
                        {"authors": {"$regex": search_term, "$options": "i"}}
                    ]
                }
            else:
                query = {}
            
            # Execute query
            cursor = collection.find(query).limit(5)
            count = 0
            async for paper in cursor:
                count += 1
                print(f"\n📄 Paper {count}:")
                print(f"  ID: {paper.get('arxiv_id', 'N/A')}")
                print(f"  Title: {paper.get('title', 'No title')}")
                print(f"  Authors: {', '.join(paper.get('authors', []))}")
                print(f"  Categories: {', '.join(paper.get('categories', []))}")
                print(f"  Submitted: {paper.get('submitted_date', 'N/A')}")
                print(f"  Status: {paper.get('status', 'N/A')}")
                if paper.get('abstract'):
                    print(f"  Abstract: {paper.get('abstract')[:200]}...")
            
            if count == 0:
                print("  No papers found matching criteria")
        
        # Close connections
        mongo_manager.close_connections()
        
    except Exception as e:
        logger.error(f"❌ Failed to query papers: {e}")

async def show_embeddings_info():
    """Show information about stored embeddings"""
    try:
        # Load configuration
        config_data = load_config()
        mongodb_config = config_data.get('mongodb', {})
        # Only use the core MongoDBConfig fields
        config = MongoDBConfig(
            connection_string=mongodb_config.get('connection_string', 'mongodb://localhost:27017/'),
            database_name=mongodb_config.get('database_name', 'arxplorer'),
            use_async=mongodb_config.get('use_async', True)
        )
        
        # Initialize MongoDB manager
        mongo_manager = MongoDBManager(config)
        await mongo_manager.initialize_database()
        
        print(f"\n🧠 EMBEDDINGS ANALYSIS")
        print("-" * 50)
        
        if config.use_async:
            collection = mongo_manager.async_db['embeddings']
            
            # Get embedding stats
            pipeline = [
                {
                    "$group": {
                        "_id": "$model_name",
                        "count": {"$sum": 1},
                        "avg_dimension": {"$avg": "$embedding_dimension"},
                        "latest": {"$max": "$created_at"}
                    }
                }
            ]
            
            async for result in collection.aggregate(pipeline):
                print(f"Model: {result['_id']}")
                print(f"  Papers: {result['count']}")
                print(f"  Dimensions: {int(result['avg_dimension'])}")
                print(f"  Latest: {result['latest']}")
        
        # Close connections
        mongo_manager.close_connections()
        
    except Exception as e:
        logger.error(f"❌ Failed to analyze embeddings: {e}")

async def main():
    """Main interactive function"""
    print("🚀 ArXplorer MongoDB Data Viewer")
    print("Choose an option:")
    print("1. Browse all collections")
    print("2. Search papers") 
    print("3. View embeddings info")
    print("4. All of the above")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            await browse_collections()
        elif choice == "2":
            search_term = input("Enter search term (or press Enter for all papers): ").strip()
            await query_papers(search_term if search_term else None)
        elif choice == "3":
            await show_embeddings_info()
        elif choice == "4":
            await browse_collections()
            await query_papers()
            await show_embeddings_info()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n\nExiting...")

if __name__ == "__main__":
    asyncio.run(main())