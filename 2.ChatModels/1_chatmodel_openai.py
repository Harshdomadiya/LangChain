from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model =  ChatOpenAI(model='gpt-4',temperature=0,max_completion_tokens=10) #temp :- 0,1.8,  for determistic task =0 , creative task = above 1

result = model.invoke("What is the capital of India")

print(result.content)