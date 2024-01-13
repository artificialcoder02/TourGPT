from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os

from dotenv import load_dotenv
import chainlit as cl

# Load environment variables from .env file
load_dotenv()
  prompt_template = """
    Your name is TOURGPT , you help in planning trips with help of the additional information from the provided context. If incase you are unsure about any detail from the provided context you might ask that, otherwise you may ask if you need any more details. Make sure to add emojis and make the trip itinerary user-friendly.Make sure you ask your questions only after you get the data from the context. \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

HUGGINGFACEHUB_API_TOKEN = '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'

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.7, "max_new_tokens":2000})


template = """ Help me plan a trip to [destination]. Include recommendations for accommodations, dining options (especially [specific dietary preferences]), key attractions and activities suitable for [type of traveler], transportation advice, budget-friendly tips, and any cultural or safety considerations. Also, how can I contribute to sustainable tourism in [destination]? Provide a comprehensive itinerary for [duration of trip]. {question}

Answer: Let's think step by step."""


@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # Send the response
    await cl.Message(content=res["text"]).send()