"""Cookbook 03: Knowledge Graph RAG — entity extraction + graph search + fusion.

Shows how to build a knowledge graph from documents, search it for
multi-hop relationships, and fuse graph results with vector retrieval.
"""

from ragpipe.graph import KnowledgeGraph, Triple

# 1. Build a knowledge graph
kg = KnowledgeGraph()

# Add triples manually (or use kg.add_document() with LLM extraction)
kg.add_triples([
    Triple(subject="Python", predicate="created_by", object="Guido van Rossum"),
    Triple(subject="Guido van Rossum", predicate="born_in", object="Netherlands"),
    Triple(subject="Netherlands", predicate="located_in", object="Europe"),
    Triple(subject="Python", predicate="used_for", object="Machine Learning"),
    Triple(subject="Machine Learning", predicate="subfield_of", object="Artificial Intelligence"),
    Triple(subject="Artificial Intelligence", predicate="researched_at", object="Stanford"),
    Triple(subject="PyTorch", predicate="written_in", object="Python"),
    Triple(subject="PyTorch", predicate="created_by", object="Meta AI"),
    Triple(subject="TensorFlow", predicate="written_in", object="Python"),
    Triple(subject="TensorFlow", predicate="created_by", object="Google"),
])

print(kg.summary())

# 2. Heuristic extraction from text (no LLM needed)
triples = kg.add_document(
    "Rust is a systems programming language. Rust was created by Mozilla. "
    "Mozilla is located in San Francisco.",
    source="rust_intro.md",
)
print(f"\nExtracted {len(triples)} triples from text")

# 3. Graph search — multi-hop traversal
print("\n--- Search: 'Python' (2 hops) ---")
results = kg.search("Python", max_hops=2)
for r in results:
    print(f"  Entity: {r.entity} (score={r.score:.2f}, hops={r.hops})")
    for rel in r.related_entities[:5]:
        print(f"    → {rel['entity']} via '{rel['relation']}' ({rel['hops']} hop)")

# 4. Get direct neighbors
print("\n--- Neighbors of 'Python' ---")
for n in kg.get_neighbors("Python"):
    print(f"  {n['entity']}: {n['relations']}")

# 5. Entity lookup
entity = kg.get_entity("Machine Learning")
if entity:
    print(f"\nEntity: {entity.name} (mentions={entity.mentions})")

# 6. Graph stats
print(f"\nGraph: {kg.entity_count} entities, {kg.triple_count} triples")
