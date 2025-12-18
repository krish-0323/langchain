from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv

import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.header('Research Tool')

paper_input = st.selectbox("Select Research Paper Name", ["Attention is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are few shot learners", "Diffusions Models Beat GANs on Image Synthesis", "Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper"])

style_input = st.selectbox("Select Explanation Style", ["Beginner Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 parapraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

template = load_prompt('template.json')



if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    # fill the placeholders

    # prompt = template.invoke({
    # })
    # result = model.invoke(prompt)
    st.write(result.content)
    # st.text('Some random text')
