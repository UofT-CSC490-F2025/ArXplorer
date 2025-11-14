import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import cProfile
import pstats
import asyncio
from datetime import datetime

# ====== IMPORT PROJECT MODULES ======
from src.core.pipeline import EmbeddingGenerator, VectorIndexer
from src.core.schemas import PipelineConfig, ProcessedPaper
from src.data.mongodb.operations import PaperRepository
from src.data.mongodb.client import MongoDBClient


# ====== HELPER: RUN & PRINT PROFILE ======
def run_profile(label: str, func_str: str):
    print(f"\n\n========== PROFILING: {label} ==========")
    profile_file = f"profile_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"

    cProfile.run(func_str, profile_file)

    p = pstats.Stats(profile_file)
    p.sort_stats("tottime").print_stats(20)


# ====== 1) _encode_text() TEST ======
def profile_encode_text():
    config = PipelineConfig(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    gen = EmbeddingGenerator(config)
    text = "Deep learning approaches for federated optimization."
    return gen._encode_text(text)



# ====== 2) generate_embeddings() TEST ======
def profile_generate_embeddings():
    config = PipelineConfig(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    gen = EmbeddingGenerator(config)

    paper = ProcessedPaper(
        arxiv_id="1234.5678",
        cleaned_title="Deep Learning for NLP",
        cleaned_abstract="Transformer models for classification tasks...",
        extracted_keywords=["deep", "learning"],
        word_count=120,
        readability_score=12.5
    )
    return gen.generate_embeddings(paper)



# ====== 3) build_index() TEST ======
def profile_build_index():
    # Create a proper PipelineConfig
    config = PipelineConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        faiss_index_type="FLAT",     # Flat index for simplicity
        n_clusters=8
    )

    gen = EmbeddingGenerator(config)
    indexer = VectorIndexer(config)

    # generate dummy embeddings
    papers = []
    for i in range(50):  # small batch for test
        p = ProcessedPaper(
            arxiv_id=str(i),
            cleaned_title="Sample Title",
            cleaned_abstract="Sample abstract for FAISS indexing test.",
            extracted_keywords=["sample"],
            word_count=50,
            readability_score=10.2
        )
        emb = gen.generate_embeddings(p)
        papers.append(emb)

    indexer.build_index(papers)



# ====== 4) search_papers_by_text() TEST ======
async def profile_search_by_text():
    # Fake repository to avoid real MongoDB connection
    class FakeRepo:
        async def search_papers_by_text(self, text_query: str, limit: int = 50):
            # pretend that DB returned 10 results
            return [{"arxiv_id": "FAKE-ID", "title": "Fake Paper"} for _ in range(limit)]

    fake_repo = FakeRepo()
    return await fake_repo.search_papers_by_text("machine learning", limit=10)



# ====== 5) search_papers_by_category() TEST ======
async def profile_search_by_category():
    class FakeRepo:
        async def get_papers_by_categories(self, categories, limit=50):
            return [{"arxiv_id": "FAKE-ID", "category": categories[0]} for _ in range(limit)]

    fake_repo = FakeRepo()
    return await fake_repo.get_papers_by_categories(["cs.LG"], limit=10)



# ====== MAIN RUNNER ======

if __name__ == "__main__":

    # 1) _encode_text
    run_profile("encode_text", "profile_encode_text()")

    # 2) generate_embeddings
    run_profile("generate_embeddings", "profile_generate_embeddings()")

    # 3) build_index
    run_profile("build_index", "profile_build_index()")

    # 4) search_papers_by_text
    run_profile("search_text", "asyncio.run(profile_search_by_text())")

    # 5) search_papers_by_category
    run_profile("search_category", "asyncio.run(profile_search_by_category())")
