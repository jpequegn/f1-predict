"""Chat interface page for F1 Race Predictor web app."""

import uuid

import streamlit as st


def show_chat_page() -> None:
    """Display the LLM-powered chat interface."""
    st.title("💬 F1 Chat Assistant")

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        st.session_state.conversation_id = str(uuid.uuid4())

    # Chat container with history
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Display attachments if any
                if "attachments" in message and message["attachments"]:
                    for attachment in message["attachments"]:
                        if attachment["type"] == "metric":
                            st.metric(
                                attachment.get("label", ""),
                                attachment.get("value", ""),
                            )
                        elif attachment["type"] == "table":
                            st.dataframe(attachment["data"], use_container_width=True)

    st.markdown("---")

    # Chat input
    col1, col2 = st.columns([6, 1])

    with col1:
        prompt = st.chat_input(
            "Ask about F1 predictions, statistics, or race analysis..."
        )

    if prompt:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            response, attachments = generate_chat_response(
                prompt,
                st.session_state.conversation_id,
            )

            st.markdown(response)

            # Display attachments
            if attachments:
                for attachment in attachments:
                    if attachment["type"] == "metric":
                        st.metric(
                            attachment.get("label", ""),
                            attachment.get("value", ""),
                        )
                    elif attachment["type"] == "table":
                        st.dataframe(attachment["data"], use_container_width=True)

            # Add assistant message
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "attachments": attachments,
                }
            )

        # Rerun to update chat display
        st.rerun()

    # Sidebar with suggested queries
    with st.sidebar:
        st.markdown("### 💡 Suggested Queries")

        suggestions = [
            "Who will win the next race?",
            "Compare Max Verstappen and Lewis Hamilton",
            "Show Red Bull's 2024 season performance",
            "What's the weather forecast for Monaco?",
            "Analyze the current championship standings",
            "Which driver has the best qualifying record?",
        ]

        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{hash(suggestion)}"):
                # Add suggestion to chat
                st.session_state.chat_messages.append(
                    {"role": "user", "content": suggestion}
                )
                st.rerun()

        # Settings
        st.markdown("---")
        st.markdown("### ⚙️ Chat Settings")

        st.selectbox(
            "AI Model",
            options=["GPT-4", "Claude 3", "Local LLM"],
            index=1,
        )

        st.slider(
            "Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
        )

        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()


def generate_chat_response(
    prompt: str,
    conversation_id: str,
) -> tuple[str, list]:
    """Generate AI response to chat prompt.

    Args:
        prompt: User's chat message
        conversation_id: Unique conversation ID for context

    Returns:
        Tuple of (response_text, attachments_list)
    """
    # Mock responses based on prompts
    prompt_lower = prompt.lower()

    attachments = []

    # Mock responses for different queries
    if "win" in prompt_lower and "race" in prompt_lower:
        response = (
            "Based on recent form and historical data, Max Verstappen has a 65% "
            "probability of winning the next race. Lewis Hamilton follows with 45% "
            "and Charles Leclerc at 38%. These predictions factor in current car "
            "performance, driver form, and circuit characteristics."
        )
        attachments = [
            {
                "type": "metric",
                "label": "Max Verstappen",
                "value": "65%",
            },
            {
                "type": "metric",
                "label": "Lewis Hamilton",
                "value": "45%",
            },
        ]

    elif "compare" in prompt_lower:
        response = (
            "Based on 2024 season statistics:\n\n"
            "**Max Verstappen:** 15 wins, 87.2 avg points per race\n"
            "**Lewis Hamilton:** 8 wins, 72.5 avg points per race\n\n"
            "Verstappen has a significant advantage in both wins and consistency "
            "this season. His race pace is particularly strong on high-speed circuits."
        )

    elif "red bull" in prompt_lower or "team" in prompt_lower:
        response = (
            "Red Bull Racing is currently leading the 2024 Constructor Championship "
            "with 512 points. Key strengths:\n"
            "- Exceptional high-speed performance\n"
            "- Strategic pit stop execution\n"
            "- Strong driver lineup consistency\n\n"
            "Their car development has focused on aerodynamic efficiency, "
            "giving them an edge in dry conditions."
        )

    elif "weather" in prompt_lower or "forecast" in prompt_lower:
        response = (
            "Weather can significantly impact race outcomes. For circuits like "
            "Monaco or Singapore, unpredictability is higher. Current forecasts "
            "suggest stable conditions for the upcoming races, favoring consistent "
            "performers like Mercedes and Red Bull."
        )

    elif "standings" in prompt_lower or "championship" in prompt_lower:
        response = "Current 2024 Championship Standings:"
        attachments = [
            {
                "type": "table",
                "data": {
                    "Position": [1, 2, 3, 4, 5],
                    "Driver": [
                        "Max Verstappen",
                        "Lewis Hamilton",
                        "Charles Leclerc",
                        "Lando Norris",
                        "Carlos Sainz",
                    ],
                    "Points": [310, 275, 245, 210, 195],
                    "Wins": [15, 8, 5, 3, 2],
                },
            }
        ]

    elif "qualify" in prompt_lower:
        response = (
            "Qualifying performance analysis shows Max Verstappen leads with "
            "15 pole positions in 2024. His qualifying pace is particularly strong "
            "on street circuits (Monaco, Singapore) and technical layouts (Hungary, "
            "Silverstone). Lewis Hamilton averages pole positions on power circuits "
            "like Monza and high-speed corners."
        )

    else:
        response = (
            "I can help you with F1 predictions, statistics, driver comparisons, "
            "and race analysis. Feel free to ask about:\n\n"
            "- **Predictions:** Who will win the next race?\n"
            "- **Comparisons:** Compare any two drivers or teams\n"
            "- **Statistics:** Championship standings, historical records\n"
            "- **Analysis:** Why a driver performs better at certain circuits\n"
            "- **Weather:** How weather impacts races\n\n"
            "What would you like to know?"
        )

    return response, attachments
