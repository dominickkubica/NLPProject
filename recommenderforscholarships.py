# -*- coding: utf-8 -*-
"""SCU Scholarship Finder"""

import streamlit as st
from datetime import datetime
from scholarship_pipeline import run_pipeline

# Configure the page
st.set_page_config(
    page_title="SCU Scholarship Finder",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("📚 Navigation")
nav_option = st.sidebar.radio("Go to:", ["🏠 Home", "🎓 Find Scholarships", "📊 Statistics", "ℹ️ About"])

# Home Page
if nav_option == "🏠 Home":
    st.title("🎓 Welcome to SCU Scholarship Finder!")
    
    # Personalized greeting based on time of day
    st.subheader("Hello!👋")

    # Introductory Text
    st.markdown("""
    **Discover scholarships tailored for Santa Clara University students.**  
    Use this platform to explore funding opportunities, get personalized recommendations, and plan for upcoming deadlines.
    """)

    # Quick Links Section
    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - [SCU Financial Aid Office](https://www.scu.edu/financialaid/)
    - [Scholarship Application Tips](https://www.scu.edu/globalengagement/study-abroad/get-started/affording-study-abroad/apply-to-scholarships/#:~:text=Scholarships%20can%20be%20local%2C%20regional,application%20as%20a%20starting%20place.)
    - [SCU Career Center](https://www.scu.edu/careercenter/)
    """)

    # Scholarship Tips Section
    st.markdown("### 💡 Scholarship Tips")
    st.markdown("""
    - **Start Early**: Begin your search and application process well in advance of deadlines.
    - **Tailor Your Applications**: Customize essays and responses to match each scholarship's requirements.
    - **Leverage SCU Resources**: Reach out to the financial aid office or academic advisors for guidance.
    """)

    # Upcoming Deadlines
    st.markdown("### 📅 Upcoming Deadlines")
    st.markdown("""
    - **SCU Merit Scholarship**: December 15, 2024  
    - **Diversity in Tech Award**: December 20, 2024  
    - **Graduate Assistantship Grant**: January 10, 2025  
    """)

    # FAQs Section
    st.markdown("### ❓ FAQs")
    st.markdown("""
    - **Who can apply for scholarships?**  
      Most scholarships are available to SCU students who meet specific criteria, such as academic performance or financial need.
    - **Do I need to file FAFSA?**  
      Filing FAFSA is required for need-based scholarships and federal aid.
    - **Where can I get help with my application?**  
      Visit the [SCU Financial Aid Office](https://www.scu.edu/financialaid/) or contact your academic advisor.
    """)

    st.balloons()

# Scholarship Finder Page
elif nav_option == "🎓 Find Scholarships":
    st.header("🎓 Find Scholarships")

    # Collect user input
    user_query = st.text_area("Describe the type of scholarship you're looking for (e.g., major, GPA, financial need):")
    
    # Submit and display results
    if st.button("🔍 Find Scholarships"):
        with st.spinner("Searching for scholarships..."):
            search_results = run_pipeline(user_query)
        
        # Display results
        if isinstance(search_results, pd.DataFrame) and not search_results.empty:
            st.subheader("Search Results")
            st.dataframe(search_results)
        else:
            st.subheader("No Results Found")
            st.markdown("Sorry, no scholarships matched your query. Please try a different description.")

# Statistics Page
elif nav_option == "📊 Statistics":
    st.header("📊 Scholarship Statistics")
    st.markdown("Explore trends and insights related to SCU scholarships.")

    # Placeholder for simple text-based statistics
    st.markdown("""
    - **Merit-Based Scholarships**: 30% of total scholarships.
    - **Need-Based Scholarships**: 20%.
    - **Diversity Scholarships**: 15%.
    - **Graduate Aid**: 10%.
    """)

# About Page
elif nav_option == "ℹ️ About":
    st.header("ℹ️ About This App")
    st.markdown("""
    **SCU Scholarship Finder** is designed to assist Santa Clara University students in finding and applying for scholarships.

    ### Features:
    - Explore SCU-specific scholarships.
    - View simple statistics on funding opportunities.
    - Receive tailored recommendations based on your profile.

    Built with ❤️ for SCU students.
    """)
    st.markdown("[Visit SCU Financial Aid Office](https://www.scu.edu/financial-aid/)")
