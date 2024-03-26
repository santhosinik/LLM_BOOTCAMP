import os
import llm_commons
from llm_commons.proxy.base import set_proxy_version
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import langchain.embeddings.openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
import streamlit as st
from ipywidgets import widgets
import json
import os
import llm_commons.proxy.base
from llm_commons.proxy.identity import AICoreProxyClient
from llm_commons.langchain.proxy import ChatOpenAI
from llm_commons.proxy.base import set_proxy_version
from llm_commons.langchain.proxy import init_llm, init_embedding_model


cf_creds_file = widgets.Text(
    value='config.json',  # service credentials file path
    placeholder="Path to service instance's credential file",
    description='',
    disabled=False
)

resource_group = widgets.Text(
    value='47d7157c-af7b-4416-816c-5c0871523092',  # resource group
    placeholder='Resource group of deployments',
    description='',
    disabled=False
)

with open(cf_creds_file.value) as cf:
    credCF = json.load(cf)

base_url = credCF["serviceurls"]["AI_API_URL"] + "/v2",  # The present SAP AI Core API version is 2
auth_url = credCF["url"] + "/oauth/token",  # Suffix to add
client_id = credCF['clientid'],
client_secret = credCF['clientsecret']

os.environ['AICORE_LLM_AUTH_URL'] = auth_url[0]
os.environ['AICORE_LLM_CLIENT_ID'] = client_id[0]
os.environ['AICORE_LLM_CLIENT_SECRET'] = client_secret
os.environ['AICORE_LLM_API_BASE'] = base_url[0]
os.environ['AICORE_LLM_RESOURCE_GROUP'] = resource_group.value
os.environ['LLM_COMMONS_PROXY'] = 'aicore'

llm_commons.proxy.resource_group = os.environ['AICORE_LLM_RESOURCE_GROUP']
llm_commons.proxy.api_base = os.environ['AICORE_LLM_API_BASE']
llm_commons.proxy.auth_url = os.environ['AICORE_LLM_AUTH_URL']
llm_commons.proxy.client_id = os.environ['AICORE_LLM_CLIENT_ID']
llm_commons.proxy.client_secret = os.environ['AICORE_LLM_CLIENT_SECRET']

aic_proxy_client = AICoreProxyClient()
set_proxy_version('aicore')  # for an AI Core proxy
from llm_commons.proxy.base import set_proxy_version
set_proxy_version('aicore') # for an AI Core proxy
aic_proxy_client.get_deployments()
llm = ChatOpenAI(proxy_model_name='gpt-4')
# Function to read PDF content
def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Mapping of PDFs
pdf_mapping = {
    'GenAI FAQ': 'SAP AI Ethics Handbook.pdf',
    'SAP Business AI': 'SAP_Business_AI_Whitepaper.pdf',
    # Add more mappings as needed
}

# Load environment variables
load_dotenv()

# Main Streamlit app
def main():
    st.title("Knowledge Assistant")
    with st.sidebar:
        st.title('🔍 Search in PDF')
        st.markdown('''
        ## About
        Choose the desired PDF, then perform a query.
        ''')


    custom_names = list(pdf_mapping.keys())

    selected_custom_name = st.sidebar.selectbox('Choose your PDF', ['', *custom_names])

    selected_actual_name = pdf_mapping.get(selected_custom_name)

    if selected_actual_name:
        pdf_folder = "pdfs"
        file_path = os.path.join(pdf_folder, selected_actual_name)

        try:
            text = read_pdf(file_path)
            st.info("The content of the PDF is hidden. Type your query in the chat window.")
        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return
        except Exception as e:
            st.error(f"Error occurred while reading the PDF: {e}")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        # Process the PDF text and create the documents list
        documents = text_splitter.split_text(text=text)

        # Vectorize the documents and create vectorstore
        embeddings = init_embedding_model('text-embedding-ada-002', proxy_client=aic_proxy_client,
                                         deployment_id='d67e66aa880020a1', api_base=llm_commons.proxy.api_base)
        vectorstore = FAISS.from_texts(documents, embedding=embeddings)

        st.session_state.processed_data = {
            "document_chunks": documents,
            "vectorstore": vectorstore,
        }
        # Save vectorstore using pickle
        pickle_folder = "Pickle"
        if not os.path.exists(pickle_folder):
            os.mkdir(pickle_folder)

        pickle_file_path = os.path.join(pickle_folder, f"{selected_custom_name}.pkl")

        if not os.path.exists(pickle_file_path):
            with open(pickle_file_path, "wb") as f:
                pickle.dump(vectorstore, f)

        # Load the Langchain chatbot
        #llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your questions from PDF "f'{selected_custom_name}'"?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
            print(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = result["answer"]
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
            print(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()