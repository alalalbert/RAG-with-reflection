import ast
import json
import os
import warnings
from typing import Any, Dict, List, Tuple
import re
import cohere
import ebooklib
import fitz
import markdown
import pandas as pd
from canopy.chat_engine import ChatEngine
from canopy.context_engine import ContextEngine
from canopy.knowledge_base import KnowledgeBase, list_canopy_indexes
from canopy.models.data_models import (
  AssistantMessage,
  Document,
  Messages,
  Query,
  UserMessage,
)
from canopy.tokenizer import Tokenizer
from ebooklib import epub
from IPython.display import HTML, display
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from tqdm.auto import tqdm

#check/initialize environment
'''
needed_env_vars = ["PINECONE_API_KEY",
                   "OPENAI_API_KEY",
                   "SERPAPI_API_KEY",
                   "COHERE_API_KEY"]

def check_and_request_env_var(var_name):
  if os.environ.get(var_name) is None:
    # Environment variable is not set. Ask the user for its value.
    print(f"{var_name} is not set. Please provide its value.")
    os.environ[var_name] = getpass.getpass(f"Enter {var_name}: ")
  else:
    # Environment variable is already set. 
    print(f"{var_name} is set.")

for var_name in needed_env_vars:
  check_and_request_env_var(var_name)
  
# 
'''

# adding docs to context, if any

def remove_markdown(content):
# Patterns to remove (simple cases)
  patterns = {
      r'\*\*(.*?)\*\*': r'\1',  # Bold
      r'\*(.*?)\*': r'\1',      # Italic
      r'\!\[(.*?)\]\((.*?)\)': r'',  # Images
      r'\[(.*?)\]\((.*?)\)': r'\1',  # Links
      r'\#\s*(.*?)\n': r'\1\n',      # Headers
      r'\>\s*(.*?)\n': r'\1\n'       # Blockquotes
  }
  # Apply each regex pattern to the content
  for pattern, repl in patterns.items():
      content = re.sub(pattern, repl, content)
  # Additional clean up for any residual Markdown symbols
  content = re.sub(r'[\`\>\#\*\_\!\[\]]', '', content)
  
  return content

def extract_text_from_epub(file_path):
  book = epub.read_epub(file_path)
  text = []
  for item in book.get_items():
      if item.get_type() == ebooklib.ITEM_DOCUMENT:
          try:
              content = item.get_content().decode('utf-8')
              plain_text = remove_markdown(content)
              text.append(plain_text)
          except FileNotFoundError:
              print(f"File {item.get_href()} not found in EPUB, skipping.")
  return "\n".join(text)

def extract_text_from_pdf(file_path):
  doc = fitz.open(file_path)
  text = []
  for page in doc:
      text.append(page.get_text())
  return "\n".join(text) #add something here to remove markdown

def extract_data(sources_dir):
  data = {'id': [], 'text': []}
  for id in os.listdir(sources_dir):
      file_path = os.path.join(sources_dir, id)
      if id.endswith('.epub'):
          text_data = extract_text_from_epub(file_path)
      elif id.endswith('.pdf'):
          text_data = extract_text_from_pdf(file_path)
      else:
          continue  # Skip files that are not PDF or EPUB
      data['id'].append(id)
      data['text'].append(text_data)
  return pd.DataFrame(data)


# add here something to pick up the core query - what do you want me to write an essay about?



query = "Tell me somethign interesting about mathematical proofs." #base query

#print("What do you want me to write about?")
#query = input()

#invoking model b/c we need to add something to kb if there aren't any docs to work off of
model = ChatOpenAI(model="gpt-4")
initial_prompt = ChatPromptTemplate.from_template("You are a helpful assistant that can write essays. Write a detailed, insightful essay about {query}.")
initial_chain = initial_prompt | model
initial_response = initial_chain.invoke({"query": query})

print("initial_response: ")
print(type(initial_response.content))
print(initial_response)
df = pd.DataFrame({'id' : [query], 'text' : [initial_response.content]})

while True:
  print("Do you have documents in a folder to add to context (Y/N)?")
  documents_in_folder = input()

  if documents_in_folder.lower() == "y":
      warnings.filterwarnings('ignore')
      sources_dir = 'sources'
      df = df.append(extract_data(sources_dir), ignore_index=True)
      print("\nGreat, found these documents:\n")
      for index, element in enumerate(df.iloc[:, 0], start=1):
          print(f"[{index}] {element}\n")
      break  # Exit the loop after successful execution
  elif documents_in_folder.lower() == "n":
      print("\nOk, no docs to be added to context.\n") #bkmk - where should this go? -- a lot of downstream functionality needs to be ripped out if we get a "no" here
      break  # Exit the loop
  else:
      
      print("\nInvalid input. Please enter Y or N.\n")
      # Loop will continue until a valid input is provided

Tokenizer.initialize()
INDEX_NAME = "advanced-rag"
kb = KnowledgeBase(index_name=INDEX_NAME)

if not any(name.endswith(INDEX_NAME) for name in list_canopy_indexes()):
    kb.create_canopy_index()
documents = [Document(**row) for _, row in df.iterrows()]

kb.connect()

batch_size = 10
for i in tqdm(range(0, len(documents), batch_size)):
    kb.upsert(documents[i: i+batch_size])

context_engine = ContextEngine(kb)
chat_engine = ChatEngine(context_engine)

def chat(new_message: str, history: Messages) -> Tuple[str, Messages]:
  messages = history + [UserMessage(content=new_message)]
  response = chat_engine.chat(messages)
  assistant_response = response.choices[0].message.content
  return assistant_response, messages + [AssistantMessage(content=assistant_response)]

def str_to_json(s):
  #print("str_to_json attempt, s:")
  #print(s)
  #print(f"Type of 's': {type(s)}")
  if isinstance(s, str):
      try:
          # Try parsing as JSON first
          return json.loads(s)
      except json.JSONDecodeError as json_error:
          print(f"Failed to parse as JSON: {json_error}")
          try:
              # If JSON parsing fails, try parsing as a Python literal
              return ast.literal_eval(s)
          except (ValueError, SyntaxError) as literal_error:
              print(f"Failed to parse as Python literal: {literal_error}")
  else:
      print("Input is not a string, attempting to convert non-string input directly.")
      return s  # If input is not a string, return it directly (might already be a list or dict)

  # If all parsing fails or input is not a string, return None
  print("Input string is neither valid JSON nor a valid Python literal.")
  return None


def evaluate_with_llm(model, prompt, generated_text):
  """
  Uses a Large Language Model (LLM) to evaluate generated text.

  :param model: An instance of the LLM, ready to generate responses.
  :param prompt: The original prompt given to the system.
  :param generated_text: The text generated by the SELF-RAG system.
  :return: A dictionary containing critique scores or assessments.
  """
  evaluations = {}
  def create_evaluation_query(template, **kwargs):
    query = ChatPromptTemplate.from_template(template)
    chain = query | model
    return float(chain.invoke(kwargs).content)
  
  # Evaluate Relevance
  relevance_template = "Given the context provided by the following prompt: '{prompt}', please evaluate on a scale from 0 to 1, where 1 is highly relevant and 0 is not relevant at all, how relevant is this generated response: '{generated_text}'? PROVIDE A NUMERICAL SCORE ONLY. NOTHING ELSE!!!"
  evaluations['relevance'] = create_evaluation_query(relevance_template, prompt=prompt, generated_text=generated_text)

  # Evaluate Clarity
  clarity_template = "How clear and easily understandable is this text: '{generated_text}'? Rate its clarity on a scale from 0 to 1, where 1 indicates that the text is very clear and 0 indicates that the text is very unclear. PROVIDE A NUMERICAL SCORE ONLY. NOTHING ELSE!!!"
  evaluations['clarity'] = create_evaluation_query(clarity_template, prompt=prompt, generated_text=generated_text)

  # Evaluate Coherence
  coherence_template = "On a scale from 0 to 1, with 1 being highly coherent and 0 being not coherent at all, how well do the ideas in this generated text: '{generated_text}' flow together? Consider if the text makes logical sense as a whole. PROVIDE A NUMERICAL SCORE ONLY. NOTHING ELSE!!!"
  evaluations['coherence'] = create_evaluation_query(coherence_template, prompt=prompt, generated_text=generated_text)

  # Evaluate Detail and Exhaustiveness
  detail_template = "Assessing the detail and exhaustiveness relative to the prompt '{prompt}', how thoroughly does this generated text: '{generated_text}' cover the topic? Rate on a scale from 0 to 1, where 1 is very detailed and exhaustive, and 0 is not detailed at all. PROVIDE A NUMERICAL SCORE ONLY. NOTHING ELSE!!!"
  evaluations['details'] = create_evaluation_query(detail_template, prompt=prompt, generated_text=generated_text)

  # Evaluate Suitability as an Answer
  suitability_template = "Evaluate the suitability of this generated text: '{generated_text}' as an answer to the original prompt '{prompt}'. On a scale from 0 to 1, where 1 is a perfect answer and 0 is completely unsuitable. PROVIDE A NUMERICAL SCORE ONLY. NOTHING ELSE!!!"
  evaluations['suitability'] = create_evaluation_query(suitability_template, prompt=prompt, generated_text=generated_text)

  return evaluations


def critique(model, prompt, generated_text):
  evaluation_weights = {
      'relevance': 3,
      'clarity': 1,
      'coherence': 0.5,
      'details': 1.5,
      'suitability': 2
  }
  #print ("evaluate with LLM being called. generated text =")
  #print(generated_text)
  evaluations = evaluate_with_llm(model, prompt, generated_text)
  #print("Evaluations:", evaluations)

  # Calculate the weighted sum of the evaluations
  weighted_sum = sum(evaluations[aspect] * evaluation_weights.get(aspect, 1) for aspect in evaluations)

  # Calculate the sum of weights for the aspects evaluated
  total_weight = sum(evaluation_weights.get(aspect, 1) for aspect in evaluations)

  # Calculate the weighted average of the evaluations
  weighted_average = weighted_sum / total_weight if total_weight > 0 else 0

  return [weighted_average, evaluations]


def is_retrieval_needed(model, prompt):
  is_retrieval_needed_prompt = ChatPromptTemplate.from_template("Given the prompt: '{prompt}', would retrieval from an external source be beneficial writing an essay on the question? Reply with only True or False")
  is_retrieval_needed_chain = is_retrieval_needed_prompt | model

  return is_retrieval_needed_chain.invoke({"prompt": prompt}).content


def consolidate(model, text):
  consolidate_prompt = ChatPromptTemplate.from_template("Given the following set of texts, please consolidate them: '{text}'")
  consolidate_chain = consolidate_prompt | model

  return consolidate_chain.invoke({"text": text}).content

def compare(model, query, text1, text2):
  compare_prompt = ChatPromptTemplate.from_template("Given the following query: '{query}', score text1 and text2 between 0 and 1, to indicate which provides a better answer overall to the query. Reply with two numbers in an array, for example: [0.1, 0.9]. The sum total of the values should be 1. text1: '{text1}' \n text2: '{text2}'")
  compare_chain = compare_prompt | model
  #print("str_to_json called by compare function")
  return str_to_json(compare_chain.invoke({"query": query, "text1": text1, "text2": text2}).content)


def generate_queries(model, prompt, num_queries):
  query_generation_prompt = ChatPromptTemplate.from_template("Given the prompt: '{prompt}', generate {num_queries} questions that are better articulated. Return in the form of an list. INCLUDE NOTHING BUT THE LIST!!! For example: [\"question 1\", \"question 2\", \"question 3\"]")
  query_generation_chain = query_generation_prompt | model
  #print("str_to_json called by generate_queries function")
  return str_to_json(query_generation_chain.invoke({"prompt": prompt, "num_queries": num_queries}).content)


def extract_documents_texts(results):
    # Initialize an empty list to store the extracted texts
    all_texts = []
    # Loop through each QueryResult in the results list
    for result in results:
        # Assuming result.documents is the correct way to access documents in a QueryResult
        for document in result.documents:
            # Assuming document.text is the correct way to access the text of a DocumentWithScore
            all_texts.append(document.text) #bookmark
    # Return the flat list of all texts
    return all_texts


co = cohere.Client(os.environ["COHERE_API_KEY"])


def get_reranked_result(query, top_n=1):
  matches = kb.query([Query(text=query)])
  print("get_reranked result matches:") #bkmk
  print (matches) #bkmk
  docs = extract_documents_texts(matches)
  print ("get_reranked_results, docs:") #bkmk
  print(docs) #bkmk
  print(type(docs))
  if not docs: #Check for empty docs list; causes error if none found
    print("No documents found for reranking.")
    return []
  rerank_results = co.rerank(model="rerank-english-v2.0", query=query, documents=docs, top_n=top_n, return_documents=True)
  print("get_reranked_results, rerank_results:")
  #debug code via cohere
  for idx, r in enumerate(rerank_results.results):
    print(f"Document Rank: {idx + 1}, Document Index: {r.index}")
    print(f"Document: {r.document.text}")
    print(f"Relevance Score: {r.relevance_score:.2f}")
    print("\n")
  #<end debug code>
  texts = []
  # Assuming rerank_results.results exists and contains the list of results
  if hasattr(rerank_results, 'results'):
      for rerank_result in rerank_results.results:
          # Ensure there is a document and it contains text
          if rerank_result.document and rerank_result.document.text:
              text = rerank_result.document.text
              texts.append(text)
          else:
              print("No document or text found for result", rerank_result)
  else:
      print("No results found in rerank results")

  return texts


class QueryDetail:
    def __init__(self, query: str):
        self.query = query
        self.content: List[str] = []
        self.critique_score: float = 0.0
        self.critique_details: Dict[str, Any] = {}
        self.retrieval_needed: bool = False
        self.search_needed: bool = False

    def add_response(self, model, search) -> None:
        """Process the query to add response, handle retrieval and critique."""
        nonce_prompt = ChatPromptTemplate.from_template('Write me a 2000-word essay about {query}')
        nonce_chain = nonce_prompt | model
        response = " ".join(nonce_chain.invoke({"query": self.query}).content)
        #print("add response has started")
        if is_retrieval_needed(model, self.query):
            #print("is_retrieval_needed is True")
            response = " ".join(get_reranked_result(self.query, top_n=3))
            self.retrieval_needed = True
        else:
            #print("is_retrieval_needed is False")
            self.retrieval_needed = False
        
        self.content.append(response)
        #print("add_response, which calls critique; response = ")
        #print(response)

        critique_score, critique_details = critique(model, self.query, response)
        self.critique_score = critique_score
        self.critique_details = critique_details
        self.search_needed = critique_score < 0.5

        if self.search_needed:
            self.search_and_add_results(search)

    def search_and_add_results(self, search) -> None:
        """Perform a search and process the results if critique score is low."""
        search_result_raw = search.run(self.query)
        #print("str_to_json called by search_and_add_results function")
        search_result = str_to_json(search_result_raw) or []
        self.content.extend(search_result)

class QueryProcessor:
    def __init__(self, model, search, queries: List[str]):
        self.model = model
        self.search = search
        self.queries = [QueryDetail(query) for query in queries]
        #print("Query processor initialized")
        
    def process_queries(self) -> List[QueryDetail]:
        """Process each query in the list."""
        #print("Queries being processed")
        for query_detail in self.queries:
            #print("Query detail.add_response being called")
            query_detail.add_response(self.model, self.search)
            '''
            if query_detail.search_needed:
                consolidated_response = consolidate(self.model, query_detail.content)
                query_detail.content = [consolidated_response]
                critique_score, critique_details = critique(self.model, query_detail.query, consolidated_response)
                query_detail.critique_score = critique_score
                query_detail.critique_details = critique_details
                '''
        return self.queries

def advanced_rag_query(model, query: str, num_queries: int) -> List[QueryDetail]:
    search = SerpAPIWrapper()
    initial_queries = generate_queries(model, query, num_queries)
    # Check if initial_queries is not None before proceeding
    if initial_queries is None:
      print("Error: Failed to generate initial queries. Returning empty list of QueryDetail objects.")
      return []  # Return an empty list to handle this failure gracefully
    
    # Proceed with using initial_queries safely, since we now know it's not None
    initial_queries = initial_queries[:num_queries]
    query_processor = QueryProcessor(model, search, initial_queries)
    processed_queries = query_processor.process_queries()
    return processed_queries

#main thing we're doing here

#first, we invoke the model

#model = ChatAnthropic(model='claude-3-sonnet-20240229')

#then, we do an advanced_rag_query
results = advanced_rag_query(model, query, 3) #bkmk 

#print("line 411 results:")
#print (results)

combined_content = " ".join(content for result in results for content in result.content)
#print (combined_content)
advanced_rag_results = consolidate(model, combined_content)

#print("adv_rag_critique, which calls critique, passing advanced_rag_results:")
#print(advanced_rag_results)
advanced_rag_critique = critique(model, query, advanced_rag_results)

#history = []
#rag_result, history = chat(query, history)
#rag_critique = critique(model, query, rag_result)


# Convert your Markdown content and critiques to strings, assuming they might not be strings already
final_result_html = markdown.markdown(str(advanced_rag_results))
#rag_result_html = markdown.markdown(str(rag_result))
advanced_rag_critique_html = markdown.markdown(str(advanced_rag_critique))
#rag_critique_html = markdown.markdown(str(rag_critique))

print("Advanced RAG results:")
print(advanced_rag_results)

print("\nAdvanced RAG critique:")
print(advanced_rag_critique)

