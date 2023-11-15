# 📦 Streamlit based AI news generator
```
```

This application is part of an ongoing PhD research. The main target is to create an application that can generate a trustworthy news article about either a theme or a given short story and adapt it accordingly to the user requests.

## Demo app

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-jornal.streamlit.app/)

## How it works

Given a short story (3 lines), multiple LLM calls are made with the use of LangChain, Chain of Thought and internet search. These processes build a baseline article, refine it by checking factualness, style, adding real testimonies and then delivering back the news article to the user. After that, the objective is to start a conversation about how the AI Journalist could improve the article or apply any kind of style.

