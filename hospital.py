
import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import altair as alt
import traceback
import re

# 🔐 Secure API key access
openai.api_key = st.secrets.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")

# 🧠 Initialize session state
for key in ["main_df", "history", "query_log", "fallback_log"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "log" in key or key == "history" else None

# 📁 File upload UI
def load_data_ui():
    with st.sidebar.expander("📁 Load or Upload Dataset", expanded=True):
        st.markdown("Upload your CSV or use our sample dataset.")
        if st.button("📥 Load Sample Data"):
            df = pd.read_csv("https://raw.githubusercontent.com/baheldeepti/hospital-streamlit-app/main/modified_healthcare_dataset.csv")
            st.session_state["main_df"] = df
            st.success("✅ Sample dataset loaded.")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            with st.spinner("Loading your file..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state["main_df"] = df
                    st.success("✅ Uploaded data loaded successfully.")
                    st.dataframe(df.head(10))
                except Exception as e:
                    st.error(f"Error loading file: {e}")

# 📊 Dynamic chart with tooltips + labels
def try_visualize(result):
    try:
        if isinstance(result, pd.Series):
            df_plot = result.reset_index()
            df_plot.columns = ["Category", "Value"]
        elif isinstance(result, pd.DataFrame) and result.shape[1] == 2:
            df_plot = result.reset_index(drop=True)
            df_plot.columns = ["Category", "Value"]
        elif isinstance(result, pd.DataFrame) and result.shape[1] == 1:
            df_plot = result.reset_index()
            df_plot.columns = ["Category", "Value"]
        else:
            st.warning("This result can't be visualized as a simple chart.")
            return

        df_plot = df_plot[pd.to_numeric(df_plot["Value"], errors="coerce").notna()]
        df_plot["Value"] = df_plot["Value"].astype(float)

        chart = (
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                x=alt.X("Category:N", sort="-y", title=None),
                y=alt.Y("Value:Q", title="Value"),
                tooltip=["Category", "Value"]
            )
            .properties(width=600, height=400)
        )

        text = (
            alt.Chart(df_plot)
            .mark_text(align="center", dy=-5, fontSize=12)
            .encode(x="Category:N", y="Value:Q", text=alt.Text("Value:Q", format=",.0f"))
        )

        st.altair_chart(chart + text, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not render chart: {e}")

# 📝 Summary formatter
def format_summary(summary_text: str) -> str:
    return f"📝 **Summary:** {summary_text.strip()}"

# 🧠 GPT-based summary
def get_summary(question, result_str):
    summary_prompt = f"You are a helpful assistant.\nThe user asked: {question}\nThe result of the query was: {result_str}\nSummarize the insight clearly."
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        return format_summary(response.choices[0].message.content.strip())
    except:
        return ""

# 🧠 Main chat handler
def handle_chat(question):
    df = st.session_state["main_df"]
    if df is None:
        st.warning("Please upload a dataset or load the sample.")
        return

    columns = ", ".join(df.columns)
    st.chat_message("user").write(question)
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.query_log.append(question)

    prompt = f"You are a senior data analyst working with this DataFrame: df\nAvailable columns: {columns}\nConversation so far:\n{st.session_state.history}\n\nWrite executable pandas code to answer the **last user question only**.\n- Assign output to a variable named `result`\n- Use only valid column names from the DataFrame\n- Do not include explanations or print statements\n- Only output valid Python code"

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
                st.markdown(summary)
                st.session_state.history.append({"role": "assistant", "content": summary})

    except Exception as e:
        st.session_state.fallback_log.append(question)
        st.chat_message("assistant").error(f"Error:\n{traceback.format_exc()}")

# 🪵 Logs only
def render_logs():
    st.subheader("🪵 Conversation Logs")
    query_log = st.session_state.get("query_log", [])
    if query_log:
        st.markdown("### 🔁 Most Asked Questions")
        log_df = pd.DataFrame(query_log, columns=["Query"])
        value_counts = log_df["Query"].value_counts().reset_index()
        value_counts.columns = ["Query", "Count"]
        st.dataframe(value_counts)
    else:
        st.info("No queries logged yet.")

    fallback_log = st.session_state.get("fallback_log", [])
    if fallback_log:
        st.markdown("### ⚠️ Fallback Queries (Unanswered)")
        fallback_df = pd.DataFrame(fallback_log, columns=["Query"])
        st.dataframe(fallback_df)

# 🧪 App UI
st.title("🏥 Hospital Chat Assistant")
st.markdown("Ask questions about hospital data. Get real answers with charts and code-backed insights!")
load_data_ui()

if prompt := st.chat_input("Ask a question about the hospital dataset..."):
    handle_chat(prompt)

# 🪵 Logs display
st.markdown("---")
render_logs()
