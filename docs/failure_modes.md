#what broke in naive, with real transcripts

- these are the things that we need to test and document
"broken rag" looks like this: 
    - it does not find the docs docs at all -> recall and retrieval problem 
    - it finds the wrong doc -> ranking problem 
    - it answers but with weak evidence -> grounding problem(hallucinations)