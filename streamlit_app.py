##### PACKAGES #####
import os
import openai
import re
import streamlit as st
from operator import itemgetter
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from functools import partial
from operator import itemgetter
from langchain.callbacks.manager import trace_as_chain_group
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
import langchain
from langchain.utilities import GoogleSerperAPIWrapper

## API keys
# Google search
os.environ["SERPER_API_KEY"] = "a395f76722402c3a8b434379cd3741ba8ea13c42"
# OpenAI
key = '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' #news
openai.api_key = key

# openai_api_key = st.sidebar.text_input('OpenAI API Key')

  ### Helper function ###
def list_out_of_num_list(input):
    numbered_list = re.findall(r'\d+\.\s+(.+)', input)
    return numbered_list

def generate_response(input_text):
   
    ##### BASELINE ARTICLE GENERATION and FACTUAL STATEMENTS IDENTIFICATION #####

    # LLM that is going to be used
    langchain.verbose = True
    llm = ChatOpenAI(temperature=0.2, model="gpt-4", openai_api_key = "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")

    # Search engine
    search = GoogleSerperAPIWrapper()

    # Tools to equip the LLM with
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful when you need to answer questions about current or past events. You should ask targeted questions specifying dates.",
        ),
    ]

    # Agent that only searches and provide responses (not being used right now), react+search works better -> in other agent
    # agent_with_search = initialize_agent(
    #     tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True, debug=True
    # )


    ### Prompts ###

    # Prompt to generate a 5 lines short story out of a theme (optional)
    short_story_prompt = PromptTemplate.from_template(
        """You are a Journalist of a digital newspaper. Given the theme, it is your job to write a factual short story (5 lines).

    Theme: {theme}
    Short Story:"""
    )

    # Prompt to generate a baseline article about the short story provided
    article_prompt = PromptTemplate.from_template(
        """You are a Journalist of a digital newspaper. Given the short factual story, it is your job to write a engaging and factual news article about this story.

    Short Story: {story}
    News article:"""
    )

    # Prompt to identify the factual statements contained within the generated baseline article
    factual_id_prompt = PromptTemplate.from_template(
        """You are a factual checker critic for a digital newspaper. Given the News Article, it is your job to identify the factual statements written in the article. Only if necessary, you can slightly rewrite a factual statement to include relevant information contained in the rest of the article in order to achieve a comprehensive and independent phrase.

    News Article: {article}
    Factual Statements:"""
    )


    # Definition of each step of the article + statements chain (currently inserting short story, instead of generating it)
    # story_provider = {"theme": RunnablePassthrough()} | short_story_prompt | llm | {"story": RunnablePassthrough()} 
    article_provider = article_prompt | llm | {"article": RunnablePassthrough()}
    factual_statements_provider = factual_id_prompt | llm 

    # Putting the steps together in a chain
    chain =  article_provider | {'article': itemgetter('article'), 'factual_statements': factual_statements_provider}

    # Invoking the chain with a short story
    # story = """As part of the Alexa Prize SocialBot Grand Challenge 5 (SGC5), a team of Stevens Institute of Technology graduate students found success with a socialbot that aims to develop a most human concept: friendship. NAM, or Never Alone with Me, was recently announced as the second place overall winner of the international university challenge competition, which focused on creating an Alexa skill that easily and clearly chats with users on trending topics and news for 20 minutes. The team won a $50,000 prize."""
    # story = "Migrant workers sent home $1.98 billion in October, a four-month high, as banks stepped up efforts to woo more remittance buoyed by a relaxed central bank rule on incentive, a development that is expected to give some relief to a country reeling under the foreign exchange crisis."
    story = input_text
    article_and_statements = chain.invoke({"story": story})


    ##### USING REACT (chain of thought + chain of verification) TO REVIEW THE FACTUALNESS OF EACH STATEMENT #####

    # Retrieving the generated article
    article = article_and_statements['article'].content 

    # Helper function to retrieve a list out of the string containing all numbered identified statements
    statement_list = list_out_of_num_list(article_and_statements['factual_statements'].content)


    # Prompt to provide a review on the factualness of each statement
    factual_checker_prompt  = PromptTemplate.from_template(
        """You are a factual checker for a digital newspaper. Given News Article and the Statement below, search in the internet and provide a very short review about the factualness of the statement. Use elements of the article to improve search results.
    News Article: {article}
    Statement: {statement}

    Short Review:""")

    # REACT agent equipped with search tool and llm
    agent_react = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Chain to apply the agent for a statement
    statement_checker = factual_checker_prompt | {'result': agent_react}

    # Building a list of reviews for each statement, giving as input the article and a statement
    reviews = []
    for statement in statement_list:
        reviews.append(statement_checker.invoke({'article': article, 'statement': statement}))  


    ##### REFINING THE BASELINE ARTICLE CONSIDERING ITS IDENTIFIED STATEMENTS AND REVIEWS #####

    # The first prompt in the chain just refine the grammar of the baseline article
    grammar_prompt = PromptTemplate.from_template("Refine this article for any grammatical errors:\n\n{context}")

    # Prompt constructor to first insert the baseline article and then incrementally insert a new statement+review and refine the baseline article
    document_prompt = PromptTemplate.from_template("{page_content}")
    partial_format_doc = partial(format_document, prompt=document_prompt)

    # Grammar chain
    grammatical_chain = {"context": partial_format_doc} | grammar_prompt | llm | StrOutputParser()

    # Refining prompt (considering each statement+review)
    refine_prompt = PromptTemplate.from_template(
        """Here's your baseline article: {prev_response}.

        Now refine the baseline article considering the factualness review of the below statement. If its factual, do not add anything, but if its not factual or inconclusive, take away just the statement from the News Article, adapting the remaining text.
        {context}"""
    )

    # Refining chain
    refine_chain = (
        {
            "prev_response": itemgetter("prev_response"),
            "context": lambda x: partial_format_doc(x["doc"]),
        }
        | refine_prompt
        | llm
        | StrOutputParser()
    )

    # Refining function to loop over all statements
    def refine_loop(docs):
        with trace_as_chain_group("refine loop", inputs={"input": docs}) as manager:
            refine = grammatical_chain.invoke(
                docs[0], config={"callbacks": manager, "run_name": "initial summary"}
            )
            for i, doc in enumerate(docs[1:]):
                refine = refine_chain.invoke(
                    {"prev_response": refine, "doc": doc},
                    config={"callbacks": manager, "run_name": f"refine {i}"},
                )
            manager.on_chain_end({"output": refine})
        return refine


    # Building the list that is going to be iterated by the refine_loop function,
    # the first element is the baseline article itself,
    # then each instance of statement+review
    statement_and_check = ['Statement: ' + statement + '\n' + 'Review: ' + response['result']['output']
                           for statement,response in zip(statement_list,reviews)]

    statement_and_check.insert(0, article)

    # Building a Langchain document out of the list
    from langchain.schema import Document
    docs = [
        Document(
            page_content = item
        )
        for item in statement_and_check
    ]

    # Applying the refining loop
    refined = refine_loop(docs)
    # Basic clean of \ns
    refined = re.sub('\n\n', ' ', refined)


    ##### TESTIMONY FINDER #####

    # Prompt to find two real testimonies about the subject being discussed in the article
    real_testimonies_prompt = PromptTemplate.from_template(
        """You are a Journalist of a digital newspaper. Given the News Article below, it is your job to search the internet for two relevant real testimonies about the content of the article, also output the name of the person who provided it. Review your observations to check if the testimonies are good enough.

    News Article: {article}

    Person 1:
    Testimony 1:

    Person 2:
    Testimony 2:"""
    )

    # React agent equipped with search and that returns all observations (searched results)
    agent_react_return_obs = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, return_intermediate_steps=True)

    # Testimony chain
    testimony_chain = {'article': RunnablePassthrough()} | real_testimonies_prompt | agent_react_return_obs

    # Invoking the testimony chain with the refined article as input
    testimonies = testimony_chain.invoke({"article": refined})


    ##### TESTIMONIES REFINER #####


    # Prompt to refine the testimonies themselves
    refine_testimonies_prompt = PromptTemplate.from_template(
        """You are a Journalist of a digital newspaper. Given the News Article below, the Observations found in a internet research and the Extracted Testimonies, it is your job to review the observations to check if the testimonies are good enough, you can include or exclude details as necessary to make the testimonies fully comprehensive.

    News Article: {article}

    Observations: {observations}

    Extracted testimonies: {testimonies}

    If no testimonies could be found, please just output 'no found testimonies'
    - Reviewed testimonies -

    Person 1:
    Testimony 1:

    Person 2:
    Testimony 2:"""
    )

    # All testimony observations from internet search
    testimony_observations = ''
    for i in testimonies['intermediate_steps']:
        testimony_observations += i[1] + '\n'

    # Testimonies output
    testimonies_output = testimonies['output']

    # Review testimony chain
    review_testimony_chain = ({"article": RunnablePassthrough(),
                              "observations": RunnablePassthrough(),
                              "testimonies": RunnablePassthrough()}
                              | refine_testimonies_prompt | llm | StrOutputParser())

    # Invoking the testimony chain with the refined article as input
    testimonies_refined = review_testimony_chain.invoke({"article": refined, "observations": testimony_observations, "testimonies": testimonies_output})


    ##### INTEGRATING THE TESTIMONIES INTO THE ARTICLE #####

    # Prompt to integrate testimonies into the refined article
    integrate_testimonies_prompt = PromptTemplate.from_template(
        """You are a Journalist of a digital newspaper. Given the News Article and Real Testimonies below, it is your job to integrate and combine them into the article, finding the best position to fit them in while maintaining and adapting the context.

    News Article: {article}

    Real Testimonies: {refined_testimonies}

    If there are no found testimonies to integrate, just repeat the article.
    Combined News Article:"""
    )

    # Refined testimonies output
    testimonies_refined_output = testimonies_refined

    # Integrate testimonies+article chain
    integrate_testimony_chain = ({"article": RunnablePassthrough(),
                                  "refined_testimonies": RunnablePassthrough()}
                                | integrate_testimonies_prompt | llm | StrOutputParser())

    # Invoking integrate chain
    refined_article_testimonies = integrate_testimony_chain.invoke({"article": refined, "refined_testimonies": testimonies_refined_output})


    ##### FORMATTING TITLE, PARAGRAPHS AND SECTIONS


    # Prompt to format the style of the text into News Article format
    format_prompt = PromptTemplate.from_template(
        """You are a Journalist designer of a digital newspaper. Given the News Article below, it is your job to format the text into a News Article design, giving it a title, splitting paragraphs/sections and making final revisions for repetitive informations.

    News Article: {combined_article}

    Formated News Article:"""
    )

    # Refined testimonies output
    combined_article_testimonies_output = refined_article_testimonies

    # Integrate testimonies+article chain
    format_chain = ({"combined_article": RunnablePassthrough()}
                                | format_prompt | llm | StrOutputParser())

    # Invoking integrate chain
    formated_article = format_chain.invoke({"combined_article": combined_article_testimonies_output})

    # giving back the response to the client
    st.info(formated_article)

st.title('🗞️ AI Journal')

with st.form('my_form'):
  text = st.text_area('Enter here your short story:',)
  submitted = st.form_submit_button('Submit')
#   if not openai_api_key.startswith('sk-'):
#     st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted:
    generate_response(text)
