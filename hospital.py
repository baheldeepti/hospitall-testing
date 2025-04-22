import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from io import StringIO
from IPython.display import display
import traceback


# 🔐 Set API key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📄 Load dataset once
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv"
    return pd.read_csv(url)

df = load_data()
columns = ", ".join(df.columns)

# 🔄 Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# 📊 Helper to display chart
def try_visualize(result):
    try:
        if isinstance(result, pd.Series):
            st.bar_chart(result)
        elif isinstance(result, pd.DataFrame):
            chart_data = result.set_index(result.columns[0]).iloc[:, 0]
            st.bar_chart(chart_data)
    except:
        pass

# 🧠 Ask GPT for a summary
def get_summary(question, result_str):
    summary_prompt = f"""You are a helpful assistant.
The user asked: {question}
The result of the query was: {result_str}
Summarize the insight clearly."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        return response.choices[0].message.content.strip()
    except:
        return ""

# 🧠 Main chatbot handler
def handle_chat(question):
    st.chat_message("user").write(question)
    st.session_state.history.append({"role": "user", "content": question})

    # Build prompt
    prompt = f"""You are a pandas expert working with this DataFrame: df
Available columns: {columns}
Conversation so far:
{st.session_state.history}

Write executable pandas code to answer the **last user question only**.
- Assign output to a variable named `result`
- Do not include explanations or print statements
- Only output valid Python code"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        code = response.choices[0].message.content.strip()
        st.code(code, language="python")

        local_vars = {"df": df}
        exec(code, {}, local_vars)
        result = local_vars.get("result", "No result")

        with st.chat_message("assistant"):
            if isinstance(result, (pd.Series, pd.DataFrame)):
                st.dataframe(result)
                try_visualize(result)
            else:
                st.write(str(result))

            summary = get_summary(question, str(result))
            if summary:
                st.markdown(f"**📝 Summary:** {summary}")
                st.session_state.history.append({"role": "assistant", "content": summary})

    except Exception as e:
        st.chat_message("assistant").error(f"Error:\n{traceback.format_exc()}")

# 🧪 Streamlit chat UI
st.title("🏥 Hospital Chat Assistant")
st.markdown("Ask questions about hospital data. Get real answers with charts and code-backed insights!")

if prompt := st.chat_input("Ask a question about the hospital dataset..."):
    handle_chat(prompt)
