import streamlit as st

st.header("ℹ️ About Our Team")
st.markdown("We are a team of students from the National University of Singapore Msc Business Analytics program dedicated to exploring the intersection of machine learning and cryptocurrency markets. We developed this dashboard and prediction models as part of our project for DBA5102")

team_members = [
    {
        "name": "Ng Jun Wei", "role": "B.Eng Civil Engineering", "image": "images/junwei.png",
        "description": "Jun Wei specializes in data engineering and visualization. He designed the dashboard interface and integrated the various components to ensure a seamless user experience."
    },
    {
        "name": "Tan Hua Swen", "role": "B.Sc Pharmacy", "image": "images/swen.png",
        "description": "Swen architected the backtesting engine and the econometric models. With a strong foundation in statistics and programming, he ensures the robustness of our trading strategies."
    },
    {
        "name": "Faris Yusri", "role": "B.Eng Chemical Engineering", "image": "images/faris.png",
        "description": "Faris brings a data-driven approach to problem-solving, focusing on feature engineering and data preprocessing to enhance model performance."
    },
    {
        "name": "Marcus Teo", "role": "B.Sc Mathematics", "image": "images/marcus.png",
        "description": "A mathematics major with a passion for data science, Marcus developed the neural network model and implemented advanced machine learning techniques to enhance prediction accuracy."
    },
    {
        "name": "Ng Yu Fei", "role": "B.Eng Biomedical Engineering", "image": "images/yufei.png",
        "description": "Yu Fei leverages an engineer's problem-solving approach and quantitative skills, leveraging statistical analysis to validate model assumptions and results."
    },
    {
        "name": "Rasyiqah Sahlim", "role": "B.Sc Biomedical Engineering", "image": "images/rasyiqah.png",
        "description": "Rasyiqah utilizes her strong analytics background to develop the XGBoost model, focusing on interpretability and feature importance to provide insights into model decisions."
    }
]

# Sort the list of members alphabetically by name
sorted_members = sorted(team_members, key=lambda x: x['name'])

# --- Create the First Row ---
st.write("---")
top_row_cols = st.columns(3)
for i in range(3):
    with top_row_cols[i]:
        member = sorted_members[i]
        st.image(member['image'], use_container_width=True)
        st.markdown(f"#### {member['name']}")
        st.markdown(f"**{member['role']}**")
        st.write(member['description'])

# --- Create the Second Row ---
st.write("---")
bottom_row_cols = st.columns(3)
for i in range(3):
    with bottom_row_cols[i]:
        member = sorted_members[i + 3]
        st.image(member['image'], use_container_width=True)
        st.markdown(f"#### {member['name']}")
        st.markdown(f"**{member['role']}**")
        st.write(member['description'])