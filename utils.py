
import os
import time
import openai
import litellm
import numpy as np
import nest_asyncio
from google.cloud import aiplatform
from llama_index.indices import VectaraIndex
from llama_index.llms import Gemini
from trulens_eval import Tru, LiteLLM, Feedback, TruLlama, feedback

def multimodal():
    print("multimodal")

def simple_query(query):
    api_key = os.environ["GOOGLE_AI_API_KEY"]
    project = os.environ["VERTEXAI_PROJECT"]
    location = os.environ["VERTEXAI_LOCATION"]
    openai.api_key = os.environ["OPENAI_API_KEY"]

    llm = Gemini(
        api_key=api_key,
        model="models/gemini-pro"
    )

    response = llm.complete(query)

    print(response)

    return response

    llm = Gemini(
        api_key=api_key,
        model="models/gemini-pro"
    )

    index = VectaraIndex(
        vectara_customer_id=os.environ["VECTARA_CUSTOMER_ID"],
        vectara_corpus_id=os.environ["VECTARA_CORPUS_ID"],
        vectara_api_key=os.environ["VECTARA_API_KEY"]
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        summary_enabled=False,
        vectara_query_mode="mmr",
        mmr_k=50,
        mmr_diversity_bias=0.5,
    )

    tru = Tru()
    tru.reset_database()

    nest_asyncio.apply()

    litellm.vertex_project = project 
    litellm.vertex_location = location
    litellm.api_key = api_key

    provider = LiteLLM(model_engine="gemini-pro")

    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on_input_output()

    context_selection = TruLlama.select_source_nodes().node.text

    f_qs_relevance = (
        Feedback(provider.qs_relevance,
                name="Context Relevance")
        .on_input()
        .on(context_selection)
        .aggregate(np.mean)
    ) 

    tru_recorder = TruLlama(
        query_engine,
        app_id="vertex",
        feedbacks=[
            f_qa_relevance,
        ]
    )

    eval_questions = [query]
    # eval_questions = ['How to know my innerself', 'Does some certain DNA have negative behaiviour?', 'how to unlock my higher purpose hidden in my DNA']
    # print(eval_questions)

    for question in eval_questions:
        with tru_recorder as recording:
            query_engine.query(question)
            time.sleep(60)

    records, feedback = tru.get_records_and_feedback(app_ids=[])
    records.head()

    time.sleep(60)

    # print("slept finished")

    tru_recorder_b = TruLlama(
        query_engine,
        app_id="vertex",
        feedbacks=[
            f_qs_relevance,
        ]
    )

    for question in eval_questions:
        with tru_recorder_b as recording:
            query_engine.query(question)
            time.sleep(60)

    records, feedback = tru.get_records_and_feedback(app_ids=[])
    records.head()
