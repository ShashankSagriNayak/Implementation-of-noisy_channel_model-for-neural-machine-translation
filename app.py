import streamlit as st
import torch
import pandas as pd
import re
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sacrebleu import sentence_bleu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="English-Hindi Translation Analyzer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #6B7280;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
    }
    .score-card {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 0.375rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .score-good {
        background-color: #DCFCE7;
        color: #16A34A;
    }
    .score-medium {
        background-color: #FEF9C3;
        color: #CA8A04;
    }
    .score-low {
        background-color: #FEE2E2;
        color: #DC2626;
    }
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        text-align: center;
        font-size: 0.875rem;
        color: #6B7280;
    }
    .table-container {
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        overflow: hidden;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th {
        background-color: #F3F4F6;
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
    }
    td {
        padding: 0.75rem 1rem;
        border-top: 1px solid #E5E7EB;
    }
    tr:nth-child(even) {
        background-color: #F9FAFB;
    }
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .direct-badge {
        background-color: #6B7280;
        color: white;
    }
    .channel-badge {
        background-color: #6B7280;
        color: white;
    }
    .loading-spinner {
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all required models for translation with progress tracking in Streamlit"""
    # Create placeholders for progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Update status
    status_text.text("Loading direct model (English → Hindi)...")
    
    # Direct model: English --> Hindi
    direct_model_name = "Helsinki-NLP/opus-mt-en-hi"
    direct_tokenizer = AutoTokenizer.from_pretrained(direct_model_name)
    direct_model = AutoModelForSeq2SeqLM.from_pretrained(direct_model_name)
    direct_model.eval()
    
    # Update progress
    progress_bar.progress(33)
    status_text.text("Loading channel model (Hindi → English)...")
    
    # Channel model: Hindi --> English
    channel_model_name = "snehalyelmati/mt5-hindi-to-english"
    channel_tokenizer = AutoTokenizer.from_pretrained(channel_model_name)
    channel_model = AutoModelForSeq2SeqLM.from_pretrained(channel_model_name)
    channel_model.eval()
    
    # Update progress
    progress_bar.progress(66)
    status_text.text("Loading Hindi language model...")
    
    # Language model for Hindi
    lm_model_name = "ai4bharat/IndicBART"
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    lm_model = AutoModelForCausalLM.from_pretrained(lm_model_name)
    lm_model.eval()
    
    # Complete loading
    progress_bar.progress(100)
    status_text.text("All models loaded successfully!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return {
        'direct_model': direct_model,
        'direct_tokenizer': direct_tokenizer,
        'channel_model': channel_model,
        'channel_tokenizer': channel_tokenizer,
        'lm_model': lm_model,
        'lm_tokenizer': lm_tokenizer
    }

def compute_seq2seq_log_prob(model, tokenizer, src_text, tgt_text):
    """
    Computes an approximate total log-probability log p(tgt_text | src_text) using a sequence-to-sequence model.
    """
    # Prepare inputs
    encodings = tokenizer(src_text, return_tensors="pt")
    tgt = tokenizer(tgt_text, return_tensors="pt").input_ids
    
    with torch.no_grad():
        outputs = model(**encodings, labels=tgt)
        total_loss = outputs.loss.item() * tgt.size(1)
    
    return -total_loss

def compute_lm_log_prob(model, tokenizer, sentence):
    """
    Computes an approximate log probability log p(sentence) using a causal language model.
    """
    inputs = tokenizer(sentence, return_tensors="pt").input_ids
    
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        total_loss = outputs.loss.item() * inputs.size(1)
    
    return -total_loss

def direct_translate(src_sentence, model, tokenizer, max_length=100):
    """
    Translate using only the direct model (baseline).
    """
    inputs = tokenizer(src_sentence, return_tensors="pt")
    
    with torch.no_grad():
        translated_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    translation = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translation

def noisy_channel_decode(src_sentence, models, beam_size=5, alpha=1.0, max_length=100):
    """
    Given a source English sentence, use the direct model to generate candidate Hindi translations.
    Then, for each candidate, compute:
    
    final_score = log p(y|x) + alpha * [ log p(x|y) + log p(y) ]
    
    and return the candidate with the highest score.
    """
    direct_model = models['direct_model']
    direct_tokenizer = models['direct_tokenizer']
    channel_model = models['channel_model']
    channel_tokenizer = models['channel_tokenizer']
    lm_model = models['lm_model']
    lm_tokenizer = models['lm_tokenizer']
    
    # Step 1: Generate candidate Hindi translations using direct model (English->Hindi)
    inputs = direct_tokenizer(src_sentence, return_tensors="pt")
    
    # Use beam search and return multiple candidates
    generated_ids = direct_model.generate(
        **inputs,
        num_beams=beam_size,
        num_return_sequences=beam_size,
        max_length=max_length,
        early_stopping=True,
    )
    
    candidates = [direct_tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
    
    best_candidate = None
    best_score = float("-inf")
    all_scores = []
    
    for cand in candidates:
        # Compute the direct model score: log p(y|x)
        direct_score = compute_seq2seq_log_prob(direct_model, direct_tokenizer, src_sentence, cand)
        
        # Compute the channel model score: log p(x|y) using the reverse (Hindi->English) model.
        channel_score = compute_seq2seq_log_prob(channel_model, channel_tokenizer, cand, src_sentence)
        
        # Compute the language model score: log p(y) for the candidate Hindi sentence.
        lm_score = compute_lm_log_prob(lm_model, lm_tokenizer, cand)
        
        # Combine scores
        final_score = direct_score + alpha * (channel_score + lm_score)
        
        all_scores.append({
            "candidate": cand,
            "direct_score": direct_score,
            "channel_score": channel_score,
            "lm_score": lm_score,
            "final_score": final_score
        })
        
        if final_score > best_score:
            best_score = final_score
            best_candidate = cand
    
    # If for some reason we couldn't find a best candidate, fall back to the first one
    if best_candidate is None and candidates:
        best_candidate = candidates[0]
        
    return best_candidate, all_scores

def calculate_sentence_bleu(hypothesis, reference):
    """
    Calculate sentence-level BLEU score using sacrebleu.
    """
    if not hypothesis or not reference:
        return 0.0
    
    # Need to provide lists to sentence_bleu
    return sentence_bleu(hypothesis, [reference]).score

def visualize_scores(all_scores):
    """
    Create visualizations for the candidate scores.
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_scores)
    
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data for model component contributions bar chart
    components_df = pd.DataFrame({
        'Direct Score': df['direct_score'],
        'Channel Score': df['channel_score'],
        'LM Score': df['lm_score']
    })
    
    # Get candidate indexes for x-axis
    candidates = [f"Cand {i+1}" for i in range(len(df))]
    
    # Bar chart for final scores
    ax0 = axes[0]
    sns.barplot(x=candidates, y=df['final_score'], ax=ax0, palette='viridis')
    ax0.set_title('Final Scores for Each Translation Candidate')
    ax0.set_xlabel('Candidate')
    ax0.set_ylabel('Final Score')
    ax0.grid(True, linestyle='--', alpha=0.7)
    ax0.tick_params(axis='x', rotation=45)
    
    # Highlight the best candidate
    best_idx = df['final_score'].idxmax()
    ax0.patches[best_idx].set_facecolor('orange')
    
    # Stacked bar chart for score components
    ax1 = axes[1]
    components_df.plot(kind='bar', stacked=False, ax=ax1, rot=45)
    ax1.set_title('Score Components for Each Candidate')
    ax1.set_xlabel('Candidate')
    ax1.set_ylabel('Score Contribution')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def main():
    # App title
    st.markdown("<h1 class='main-header'>English-Hindi Translation Analyzer</h1>", unsafe_allow_html=True)
    
    # App description
    st.markdown("""
    <div class="highlight">
        This app demonstrates the comparison between direct translation and noisy channel translation for English to Hindi. 
        It shows all generated translation candidates and their respective scores, allowing you to analyze the translation quality.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for parameters
    st.sidebar.markdown("## Model Parameters")
    
    alpha = st.sidebar.slider(
        "Alpha (weight for channel model & LM)",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Controls the weight of the channel model and language model scores"
    )
    
    beam_size = st.sidebar.slider(
        "Beam Size",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of translation candidates to generate"
    )
    
    max_length = st.sidebar.slider(
        "Max Translation Length",
        min_value=50,
        max_value=200,
        value=100,
        step=10,
        help="Maximum length of the translation in tokens"
    )
    
    reference_hi = st.sidebar.text_area(
        "Reference Hindi Translation (Optional)",
        "",
        help="If you know the correct Hindi translation, enter it here to calculate BLEU scores"
    )
    
    # Load models
    with st.spinner("Loading translation models... This may take a minute."):
        models = load_models()
    
    # Input for English text
    st.markdown("<h2 class='sub-header'>Enter English Text</h2>", unsafe_allow_html=True)
    english_text = st.text_area(
        "Type or paste English text to translate",
        "The weather is very pleasant today.",
        height=100
    )
    
    # Translation button
    col1, col2 = st.columns([1, 3])
    with col1:
        translate_button = st.button("Translate", type="primary", use_container_width=True)
    with col2:
        st.markdown("")  # Empty space for layout
    
    # Process translation when button is clicked
    if translate_button and english_text:
        st.markdown("<div class='loading-spinner'>", unsafe_allow_html=True)
        with st.spinner("Translating..."):
            # Direct translation
            direct_translation = direct_translate(english_text, models['direct_model'], models['direct_tokenizer'], max_length)
            
            # Noisy channel translation with all candidate scores
            channel_translation, all_scores = noisy_channel_decode(
                english_text, models, beam_size, alpha, max_length
            )
            
            # Calculate BLEU scores if reference is provided
            direct_bleu = None
            channel_bleu = None
            
            if reference_hi:
                direct_bleu = calculate_sentence_bleu(direct_translation, reference_hi)
                channel_bleu = calculate_sentence_bleu(channel_translation, reference_hi)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display the translation results
        st.markdown("<h2 class='sub-header'>Translation Results</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        # Direct Model Results
        with col1:
            st.markdown(
                f"""
                <div class="card">
                    <h3><span class="model-badge direct-badge">Direct Model</span> Translation</h3>
                    <div class="highlight">{direct_translation}</div>
                    {f'<p>BLEU Score: <span class="score-card {"score-good" if direct_bleu and direct_bleu > 30 else "score-medium" if direct_bleu and direct_bleu > 15 else "score-low"}">{direct_bleu:.2f}</span></p>' if direct_bleu is not None else ''}
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Noisy Channel Model Results
        with col2:
            st.markdown(
                f"""
                <div class="card">
                    <h3><span class="model-badge channel-badge">Noisy Channel</span> Translation</h3>
                    <div class="highlight">{channel_translation}</div>
                    {f'<p>BLEU Score: <span class="score-card {"score-good" if channel_bleu and channel_bleu > 30 else "score-medium" if channel_bleu and channel_bleu > 15 else "score-low"}">{channel_bleu:.2f}</span></p>' if channel_bleu is not None else ''}
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Visualizations for candidate scores
        st.markdown("<h2 class='sub-header'>Translation Candidate Analysis</h2>", unsafe_allow_html=True)
        
        # Display visualizations
        fig = visualize_scores(all_scores)
        st.pyplot(fig)
        
        # Detailed table of candidates and scores
        st.markdown("<h3 class='sub-header'>All Translation Candidates</h3>", unsafe_allow_html=True)
        
        # Format the scores dataframe for display
        candidates_df = pd.DataFrame(all_scores)
        candidates_df['rank'] = candidates_df['final_score'].rank(ascending=False).astype(int)
        candidates_df = candidates_df.sort_values('rank')
        
        # Mark the selected candidate
        best_candidate = candidates_df.iloc[0]['candidate']
        
        # Create a formatted table
        st.markdown('<div class="table-container">', unsafe_allow_html=True)
        st.write(candidates_df.rename(columns={
            'candidate': 'Translation',
            'direct_score': 'Direct Score',
            'channel_score': 'Channel Score',
            'lm_score': 'LM Score',
            'final_score': 'Final Score',
            'rank': 'Rank'
        }).style.background_gradient(subset=['Final Score'], cmap='viridis')
          .format({
              'Direct Score': '{:.2f}',
              'Channel Score': '{:.2f}',
              'LM Score': '{:.2f}',
              'Final Score': '{:.2f}'
          }))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Explanation of the scoring methodology
        with st.expander("How the scoring works"):
            st.markdown("""
            ### Score Components
            
            - **Direct Score**: Log probability of the Hindi translation given the English input: log P(Hindi|English)
            - **Channel Score**: Log probability of reconstructing the original English from the Hindi: log P(English|Hindi)
            - **LM Score**: Hindi language model fluency score: log P(Hindi)
            
            ### Final Score Calculation
            
            The noisy channel model combines these scores using:
            
            **Final Score = Direct Score + Alpha × (Channel Score + LM Score)**
            
            Where Alpha is the weighting parameter that controls the influence of the channel model and language model.
            """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        English-Hindi Translation Analyzer | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()