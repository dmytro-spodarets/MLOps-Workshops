import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import murnitur
from murnitur import Guard, GuardConfig, log
from murnitur.guard import RuleSet

# Завантаження змінних середовища
MURNITUR_API_KEY = os.getenv("MURNITUR_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://t5-or-phi-model.apps.spodarets.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "DevOpsLLM")

# Ініціалізація Murnitur
murnitur.set_api_key(MURNITUR_API_KEY)
murnitur.init(project_name="MLOps", enabled_instruments=["openai", "langchain"])

config = GuardConfig(api_key=OPENAI_API_KEY, provider="openai", group="MLOps-workshop")

# Визначення наборів правил для вхідних даних
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
            "fallback": "Я не можу обробляти запити, що містять особисту ідентифікаційну інформацію 🤐",
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
            "fallback": "Вибачте, я не можу допомогти з цим запитом.",
        },
    },
]

# Визначення наборів правил для вихідних даних
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
            "fallback": "Я не можу надавати особисту ідентифікаційну інформацію 🤐",
        },
    },
]

# Функція для обробки запиту та отримання відповіді від моделі
def process_query(query: str, model: ChatOpenAI):
    try:
        # Перевірка вхідних даних
        input_check = Guard.shield({"input": query}, input_rulesets, config)
        if input_check.triggered:
            return input_check.text, True

        # Отримання відповіді від моделі
        response = model.invoke([HumanMessage(content=query)])

        # Перевірка вихідних даних
        output_check = Guard.shield({"output": response.content}, output_rulesets, config)
        if output_check.triggered:
            return output_check.text, True

        # Логування успішної взаємодії
        log("Successful_Interaction", {"input": query, "output": response.content})

        return response.content, False
    except Exception as e:
        log("Error", {"input": query, "error": str(e)})
        return f"Сталася помилка: {str(e)}", True

# Налаштування Streamlit
st.set_page_config(page_title="DevOpsLLM Чат", page_icon="💬")
st.title("DevOpsLLM Чат")

# Ініціалізація моделі
@st.cache_resource
def get_model():
    return ChatOpenAI(
        model_name=MODEL_NAME,
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY
    )

model = get_model()

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

    # Обробка запиту з використанням моделі та Murnitur
    with st.spinner("DevOpsLLM думає..."):
        response, is_warning = process_query(user_input, model)

    if is_warning:
        st.warning(response)
    else:
        # Додавання відповіді моделі до історії
        st.session_state.messages.append(AIMessage(content=response))
        st.chat_message("assistant").write(response)