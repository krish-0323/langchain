from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFaceEndpoint(
    repo_id ='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# 1st -> prompt
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_varibles=['topic']
)

# 2nd -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text./n {text}',
    input_variables=['text']
)

# prompt1 = template1.invoke({'topic': 'black hole'})
# result = model.invoke(prompt1)
# prompt2 = template2.invoke({'text': result.content})
# result = model.invoke(prompt2)
# print(result.content)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'blackhole'})
print(result)