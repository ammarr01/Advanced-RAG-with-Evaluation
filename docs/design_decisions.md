#what naive does, what advanced does, and why each choice was made
## introduction: 
    This is a case study that aims to show why traditional RAG systems fail in production, where does it go wrong exactly(with key measurable metrics) and why Advanced RAG(few design tweaks) is a better and more accurate option for you.

## why naive rag fails:
once rag systems leave demos and go into production, best practices are no longer optional - they become very crucial and can make or break your business as they start answering questions that affect customers, legal compliance, revenue, or operations.

"broken rag" looks like this: 
    - it does not find the docs docs at all -> recall and retrieval problem 
    - it finds the wrong doc -> ranking problem 
    - it answers but with weak evidence -> grounding problem(hallucinations)



## why advanced RAG: 
once rag systems leave demos and go into production, best practices are no longer optional - they become very crucial and can make or break your business as they start answering questions that affect customers, legal compliance, revenue, or operations.

so that's why naive rag systems are not enough and can cost you heavily in the long run. that's why you need to focus on high-leverage retrieval engineering techniques: advanced chunking strats, embdedding model selection, reranking techniques, query translation, hierarchical indexing, and else.



## our methodology
1. we will start by creating a naive rag system based on a mock enterprise bank data 
2. evaluate it against an eval dataset and show failure cases
3. implement advanced rag techniques 
4. show measurable improvement in key metrics

### Naive RAG design decisions:
    - format agnostic extraction -> no table extraction
    - fixed chunk size and overlap(1000)
    - no metadata for chunks 
    - RecursiveCharacterTextSplitter that splits on text and not on meaning 
    - no reranking 
    - query used verbatim 
    - only dense embeddings 
    - minimal prompt



 ### how we will evaluate our system

