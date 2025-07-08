import json
import os
import time
import faiss
import numpy as np
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
EMBEDDING_FILE = "recipe_embeddings.npy"
INDEX_FILE     = "recipe_faiss.index"
RECIPES_FILE   = "indian_cooking.jsonl"

# --- Load Recipe Data ---
@st.cache_resource
def load_data():
    with open(RECIPES_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# --- Load or Compute Embeddings ---
@st.cache_resource
def load_embeddings(recipes):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [r["prompt"] + " " + r["response"] for r in recipes]

    if os.path.exists(EMBEDDING_FILE) and os.path.exists(INDEX_FILE):
        embeddings = np.load(EMBEDDING_FILE)
        index      = faiss.read_index(INDEX_FILE)
    else:
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        np.save(EMBEDDING_FILE, embeddings)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)

    return embedder, index

# --- Retrieve Top K Recipes ---
def retrieve(query, embedder, index, recipes, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    _, ids = index.search(q_emb, top_k)
    return [recipes[i] for i in ids[0]]

# --- Streamlit App ---
st.set_page_config(page_title="🍲 Indian Recipe RAG Assistant", page_icon="🍛")
st.title("🍛 Indian chef Assistant")
st.caption("Ask a cooking question and get recipe suggestions via RAG + streaming LLM")

query = st.text_input("Your question:", placeholder="E.g. Tell me about a recipe of aloo.")

if query:
    # 1) Display loader initially
    loader_placeholder = st.empty()
    loader_placeholder.markdown("""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
        margin-bottom: 20px;
    ">
        <div style="
            border: 8px solid #f3f3f3;
            border-top: 8px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
        "></div>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </div>
    """, unsafe_allow_html=True)

    # 2) Retrieval step
    recipes = load_data()
    embedder, index = load_embeddings(recipes)
    hits = retrieve(query, embedder, index, recipes)
    context = "\n\n".join(hit["response"] for hit in hits)

    # 3) Remove loader and start thinking card
    loader_placeholder.empty()

    # 4) Stream the LLM response with thought process and final answer
    final_prompt = f"""
You are a helpful Indian cooking assistant.

Here are some recipes that might help:

{context}

User question: {query}

First, think through how you will structure the recipe based on the provided information.
Then, provide the final recipe.

Please format your response as follows:

First, your thought process on how to create the recipe.

Then, write '===FINAL ANSWER==='

Finally, provide the structured recipe with ingredients and steps.
"""
    stream = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=[{"role": "user", "content": final_prompt}],
        stream=True
    )

    # Create placeholders for the card and final answer
    card_placeholder = st.empty()
    final_answer_placeholder = st.empty()

    full_text = ""
    separator = "===FINAL ANSWER==="
    is_final_answer = False
    start_time = time.time()

    for chunk in stream:
        part = chunk.get("message", {}).get("content", "")
        full_text += part

        if not is_final_answer and separator in full_text:
            thought_process, final_answer = full_text.split(separator, 1)
            elapsed_time = int(time.time() - start_time)
            # Display thought process in card with animated scrolling and timer
            card_placeholder.markdown(f"""
            <div style="
                border: 1px solid #444;
                border-radius: 12px;
                padding: 15px;
                background: #1c1c1c;
                color: #eee;
                font-family: 'Segoe UI', sans-serif;
                box-shadow: 0 0 15px rgba(0,0,0,0.3);
                max-height: 200px;
                overflow-y: auto;
                margin-bottom: 20px;
            " id="thinking-card">
              <div style="font-size:18px;">
                🧠 <strong>Thinking complete. ({elapsed_time}s)</strong>
              </div>
              <div style="margin-top:8px; line-height:1.5;" id="thought-content">
                {thought_process.strip()}
              </div>
              <script>
                var card = document.getElementById('thinking-card');
                var content = document.getElementById('thought-content');
                content.style.scrollBehavior = 'smooth';
                content.scrollTop = content.scrollHeight;
              </script>
            </div>
            """, unsafe_allow_html=True)
            # Set flag and reset full_text to final_answer
            is_final_answer = True
            full_text = final_answer
        else:
            if not is_final_answer:
                elapsed_time = int(time.time() - start_time)
                # Display accumulating thought process in card with animated scrolling, timer, and cursor
                card_placeholder.markdown(f"""
                <div style="
                    border: 1px solid #444;
                    border-radius: 12px;
                    padding: 15px;
                    background: #1c1c1c;
                    color: #eee;
                    font-family: 'Segoe UI', sans-serif;
                    box-shadow: 0 0 15px rgba(0,0,0,0.3);
                    max-height: 200px;
                    overflow-y: auto;
                    margin-bottom: 20px;
                " id="thinking-card">
                  <div style="font-size:18px;">
                    🧠 <strong>Thinking... ({elapsed_time}s)</strong>
                  </div>
                  <div style="margin-top:8px; line-height:1.5;" id="thought-content">
                    {full_text}▌
                  </div>
                  <script>
                    var card = document.getElementById('thinking-card');
                    var content = document.getElementById('thought-content');
                    content.style.scrollBehavior = 'smooth';
                    content.scrollTop = content.scrollHeight;
                  </script>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Display accumulating final answer outside with cursor
                final_answer_placeholder.markdown(f"### 🍽️ Suggested Recipe:\n{full_text}▌", unsafe_allow_html=True)

    # After streaming, remove cursor from final answer
    if is_final_answer:
        final_answer_placeholder.markdown(f"### 🍽️ Suggested Recipe:\n{full_text.strip()}", unsafe_allow_html=True)
    else:
        # If separator was not found, display the entire text as final answer
        final_answer_placeholder.markdown(f"### 🍽️ Suggested Recipe:\n{full_text.strip()}", unsafe_allow_html=True)