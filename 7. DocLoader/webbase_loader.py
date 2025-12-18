from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt = PromptTemplate(
    template='Answer the following questions \n {question} from the following {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url = 'https://www.cricbuzz.com/live-cricket-scores/117404/ind-vs-rsa-3rd-odi-south-africa-tour-of-india-2025'
loader = WebBaseLoader(url)

docs = loader.load()

# print(len(docs))
chain = prompt | model | parser

result = chain.invoke({'question': 'Which is team is batting first?', 'text':docs[0].page_content})
print(result)