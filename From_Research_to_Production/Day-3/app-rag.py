import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
import murnitur
from murnitur import Guard, GuardConfig, log
from murnitur.guard import RuleSet
from qdrant_client import QdrantClient

# Завантаження змінних середовища
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MURNITUR_API_KEY = os.getenv("MURNITUR_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL",
                       "https://3fb2a758-19fe-4158-b88b-41f9259cdcca.europe-west3-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Ініціалізація Murnitur
murnitur.set_api_key(MURNITUR_API_KEY)
murnitur.init(project_name="MLOps", enabled_instruments=["openai", "langchain", "qdrant"])

config = GuardConfig(api_key=OPENAI_API_KEY, provider="openai", group="MLOps-workshop")

# Ініціалізація Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name="devops_docs",
    embeddings=embeddings,
)

# Логування ініціалізації Qdrant
log("Qdrant_Initialization", {
    "collection_name": "devops_docs",
    "embeddings_model": "OpenAI",
    "qdrant_url": QDRANT_URL
})

# Визначення rulesets для вхідних даних
input_rulesets: list[RuleSet] = [
    {
        "rules": [
            {
                "metric": "input_pii",
                "operator": "contains",
                "value": ["financial_info", "email", "ssn"],
            }
        ],
        "action": {
            "type": "OVERRIDE",
            "fallback": "I cannot process requests containing personal identifiable information 🤐",
        },
    },
    {
        "rules": [
            {
                "metric": "prompt_injection",
                "operator": "contains",
                "value": [
                    "simple_instruction",
                    "instruction_override",
                    "impersonation",
                    "personal_information_request"
                ],
            }
        ],
        "action": {
            "type": "OVERRIDE",
            "fallback": "Sorry, I can't help with that request.",
        },
    },
]

# Визначення rulesets для вихідних даних
output_rulesets: list[RuleSet] = [
    {
        "rules": [
            {
                "metric": "pii",
                "operator": "contains",
                "value": ["financial_info", "email", "ssn"],
            }
        ],
        "action": {
            "type": "OVERRIDE",
            "fallback": "I cannot provide personal identifiable information 🤐",
        },
    },
]


# Функція для обробки запиту та отримання відповіді від моделі з використанням RAG
def process_query(query: str, model: ChatOpenAI):
    # Перевірка вхідних даних
    input_check = Guard.shield({"input": query}, input_rulesets, config)
    if input_check.triggered:
        log("Input_Violation", {"query": query, "violation": input_check.text})
        return input_check.text, True, None

    # Логування запиту до Qdrant
    log("Qdrant_Query", {"query": query})

    # Створення ланцюжка RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # Отримання відповіді від моделі з використанням RAG
    try:
        result = qa_chain({"query": query})
        response = result['result']
        source_documents = result['source_documents']

        # Логування успішного пошуку в Qdrant
        log("Qdrant_Search_Success", {
            "query": query,
            "num_source_documents": len(source_documents)
        })
    except Exception as e:
        log("RAG_Error", {"query": query, "error": str(e)})
        return f"An error occurred: {str(e)}", True, None

    # Перевірка вихідних даних
    output_check = Guard.shield({"output": response}, output_rulesets, config)
    if output_check.triggered:
        log("Output_Violation", {"query": query, "response": response, "violation": output_check.text})
        return output_check.text, True, None

    # Логування успішної взаємодії
    log("Successful_Interaction", {"query": query, "response": response})

    return response, False, source_documents


# Налаштування Streamlit
st.set_page_config(page_title="DevOpsLLM Chat з RAG", page_icon="💬")
st.title("DevOpsLLM Chat з RAG")

# Ініціалізація моделі
model = ChatOpenAI(
    model_name="DevOpsLLM",
    openai_api_base="https://t5-or-phi-model.apps.spodarets.com/v1",
    openai_api_key="not-needed"
)

# Ініціалізація історії чату
if "messages" not in st.session_state:
    st.session_state.messages = []

# Відображення історії чату
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)

# Поле введення для нового повідомлення
user_input = st.chat_input("Введіть ваше повідомлення:")

if user_input:
    # Додавання повідомлення користувача до історії
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Обробка запиту з використанням моделі, Murnitur та RAG
    with st.spinner("DevOpsLLM думає..."):
        response, is_warning, source_documents = process_query(user_input, model)

    if is_warning:
        st.warning(response)
    else:
        # Додавання відповіді моделі до історії
        st.session_state.messages.append(AIMessage(content=response))
        st.chat_message("assistant").write(response)

        # Відображення джерел
        if source_documents:
            st.write("Джерела:")
            for doc in source_documents:
                st.write(f"- {doc.metadata.get('source', 'Невідоме джерело')}")
