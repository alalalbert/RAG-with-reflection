# RAG with reflection

Proof of concept project for testing out advanced RAG techniques like reflection, & search augmentation. Based on sample code provided by Pinecone's advanced RAG examples implementing their Canopy context engine, but with a bunch of updates for usability & robustness.

## Getting started

Add your API keys to env, give it a query or essay topic, and put any relevant docs you have into ./sources.

### Prerequisites

Plenty of libs to install:

```
fitz
markdown
cohere
ebooklib
pandas
canopysdk
langchain
langchain-community
langchain-core
langchain-openai
langchain-text-splitters
```

There's also an experimental ```init_env.py``` that's intended to help someone without the right API keys identify any they might be missing. Will need keys for Pinecone, Cohere, OpenAI, and SerpAPI. Free tier in each is enough to get going. You will end up spending some pennies though. My dev of this cost about $5 across all the services.

### What it looks like

```What do you want me to write about?```

Put in your query here

```Do you have documents in a folder to add to context (Y/N)?```

yes/no

If you have files in ./sources, this is when it grabs them and puts them into Canopy/Pinecone

You'll end up with a more polished output, and one more reflective of what is in your sources, than you'd get from OpenAI directly.

## Reflections

After playing around with this a lot I've discovered that the reflection isn't always beneficial. Reflection is grounding, and it polishes the work a lot, but it can also strip out some of the detail from the final text. I suspect this could be improved by weighting different characteristics, or dynamically assigning characteristics to weight based on user input.

The reflection system also tends to strip out anecdotes from the final text. This is either good or bad depending on the use case, but it's reflective of the sources I provided, which were mostly bland textbooks. In another experiment, where I loaded the ./sources directory with popular history books & fed a different prompt, it did include anecdotes (signaling the source material is important for the content that gets included), but the RAG system sterilized them a bit; they came off with less life than in the source documents.

Future dev might include:
- managing settings through a config file
- temperature variation at different points (e.g., higher for generation, lower for reflection)
- model output schema enforcement
- style transfer for getting output on topic & with particular style characteristics

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Ackwnoledgements

Big ups to Pinecone for the <a href='https://github.com/pinecone-io/examples/blob/master/learn/generation/better-rag/advanced-rag-with-canopy.ipynb'>sample code</a> on which this is based

If you're reading this, check out <a href='https://www.oneusefulthing.org/'>Ethan Mollick</a> who writes about how this tech is changing the world of work
