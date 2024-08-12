import streamlit as st
import validators 
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


# streamlit app
st.set_page_config(page_title="Text summarizer - YT & Websites",
                   page_icon="🦜")

st.title("🦜 Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Getting api & urls
with st.sidebar:
    groq_API=st.text_input(label="Enter Groq API Key", 
                           value="",
                           type="password")
    
generic_url=st.text_input(label="URL",
                          label_visibility="collapsed")

if groq_API:
    # Model
    llm=ChatGroq(model="Gemma-7b-It", groq_api_key=groq_API)

    # Prompt template
    prompt="""
    Provide a summary of the following content in 250 words:
    Content:{text}"""

    prompt_template=PromptTemplate(input_variables=["text"],
                                template=prompt)

    if st.button(label="Summarize"):

        # validating the API and URL
        if not groq_API.strip() or not generic_url.strip():
            st.error(body="Please enter the API & URL details ")
        elif not validators.url(generic_url):
            st.error(body="Please enter a valid URL")
        else:
            try:
                with st.spinner(text="Loading"):
                    # load pages 
                    if "youtube.com" in generic_url:
                        loader=YoutubeLoader.from_youtube_url(youtube_url=generic_url,
                                                            add_video_info=True)
                    else:
                        loader=UnstructuredURLLoader(urls=[generic_url],
                                                    ssl_verify=False,
                                                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    
                    docs=loader.load()

                    # Stuff chain for summarization
                    chain=load_summarize_chain(llm=llm,
                                            chain_type="stuff",
                                            prompt=prompt_template)
                    response=chain.invoke(docs)
                    st.success(body=response["output_text"])

            except Exception as e:
                st.error(body=f"Error: {e}")
else:
    st.warning(body="Please Enter Groq API key")
                




