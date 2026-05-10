import os
import json
from datetime import datetime
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm


# =========================
# APP CONFIG
# =========================

load_dotenv()

st.set_page_config(
    page_title="Optimum | AI Negotiation Coach",
    page_icon="🎯",
    layout="wide"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4.1-mini"

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None


# =========================
# SESSION STATE
# =========================

def init_session_state():
    defaults = {
        "messages": [],
        "simulation_started": False,
        "simulation_finished": False,
        "scorecard": None,
        "improved_answers": None,
        "final_script": None,
        "scenario": "",
        "goal": "",
        "context": "",
        "minimum_outcome": "",
        "desired_outcome": "",
        "opponent_personality": "",
        "difficulty": "Medium",
        "user_name": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =========================
# STYLING
# =========================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 20px;
        color: #555;
        margin-top: 0px;
        margin-bottom: 24px;
    }
    .metric-card {
        background-color: #f7f7f8;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e5e5e5;
    }
    .small-note {
        font-size: 13px;
        color: #777;
    }
    .cta-box {
        background-color: #101828;
        color: white;
        padding: 24px;
        border-radius: 18px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# HELPER FUNCTIONS
# =========================

def call_ai(system_prompt, user_prompt, temperature=0.7):
    if client is None:
        st.error("OpenAI API key not found. Add your API key to the .env file.")
        st.stop()

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=temperature
        )

        return response.output_text

    except Exception as e:
        st.error(f"AI request failed: {e}")
        st.stop()


def get_roleplay_system_prompt():
    return f"""
You are Optimum, an advanced AI negotiation simulator.

Your role:
You play the negotiation opponent in a realistic, high-stakes negotiation simulation.

Rules:
1. Stay in character as the other party.
2. Do not coach the user during the roleplay.
3. Push back realistically.
4. Match the selected opponent personality.
5. Keep replies concise, natural, and conversational.
6. Ask tough questions when appropriate.
7. Do not reveal that you are an AI coach.
8. Do not conclude too quickly. Let the user work for the outcome.
9. If the user becomes rude, remain professional but firm.
10. If the user makes a strong argument, acknowledge it but continue negotiating.

Scenario:
{st.session_state.scenario}

User goal:
{st.session_state.goal}

User context:
{st.session_state.context}

Minimum acceptable outcome:
{st.session_state.minimum_outcome}

Desired outcome:
{st.session_state.desired_outcome}

Opponent personality:
{st.session_state.opponent_personality}

Difficulty:
{st.session_state.difficulty}
"""


def get_conversation_text():
    conversation = ""

    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Opponent"
        conversation += f"{role}: {message['content']}\n\n"

    return conversation


def get_ai_opponent_reply():
    conversation_text = get_conversation_text()

    system_prompt = get_roleplay_system_prompt()

    user_prompt = f"""
Continue the negotiation based on the conversation so far.

Conversation:
{conversation_text}

Now respond as the negotiation opponent.
"""

    return call_ai(system_prompt, user_prompt, temperature=0.8)


def generate_scorecard():
    conversation_text = get_conversation_text()

    system_prompt = """
You are Optimum, an expert negotiation coach.

You evaluate negotiation performance with practical, direct feedback.

Return your answer in valid JSON only.

Use this exact JSON structure:
{
  "overall_score": 0,
  "confidence_score": 0,
  "clarity_score": 0,
  "anchoring_score": 0,
  "objection_handling_score": 0,
  "concession_control_score": 0,
  "empathy_score": 0,
  "closing_score": 0,
  "summary": "",
  "strengths": [],
  "weaknesses": [],
  "missed_opportunities": [],
  "next_training_focus": ""
}

Scoring:
0 means very poor.
100 means excellent.

Be honest but constructive.
"""

    user_prompt = f"""
Evaluate this negotiation.

Scenario:
{st.session_state.scenario}

User goal:
{st.session_state.goal}

Minimum acceptable outcome:
{st.session_state.minimum_outcome}

Desired outcome:
{st.session_state.desired_outcome}

Conversation:
{conversation_text}
"""

    result = call_ai(system_prompt, user_prompt, temperature=0.3)

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "overall_score": 0,
            "confidence_score": 0,
            "clarity_score": 0,
            "anchoring_score": 0,
            "objection_handling_score": 0,
            "concession_control_score": 0,
            "empathy_score": 0,
            "closing_score": 0,
            "summary": "The scorecard could not be parsed. Raw AI output is shown below.",
            "strengths": [result],
            "weaknesses": [],
            "missed_opportunities": [],
            "next_training_focus": "Try again with a shorter conversation."
        }


def generate_improved_answers():
    conversation_text = get_conversation_text()

    system_prompt = """
You are Optimum, an expert negotiation coach.

Your job is to improve the user's negotiation responses.

Return practical feedback in markdown.

For the user's weakest answers:
1. Quote or summarize the weak answer.
2. Explain why it was weak.
3. Provide a stronger version.
4. Explain why the stronger version works better.

Focus on:
- Anchoring
- Framing
- Confidence
- Objection handling
- Not conceding too early
- Asking calibrated questions
- Creating leverage
"""

    user_prompt = f"""
Improve the user's negotiation performance based on this conversation.

Scenario:
{st.session_state.scenario}

User goal:
{st.session_state.goal}

Conversation:
{conversation_text}
"""

    return call_ai(system_prompt, user_prompt, temperature=0.5)


def generate_final_script():
    conversation_text = get_conversation_text()

    system_prompt = """
You are Optimum, an expert negotiation coach.

Create a final negotiation script the user can use in the real conversation.

The script should be:
- Practical
- Confident
- Professional
- Natural
- Specific to the user's context
- Not too aggressive
- Designed to increase the chance of achieving the desired outcome

Structure:
1. Opening line
2. Value framing
3. The ask
4. Handling likely objections
5. Concession strategy
6. Closing line
7. Short version to memorize
"""

    user_prompt = f"""
Create a personalized negotiation script.

Scenario:
{st.session_state.scenario}

User goal:
{st.session_state.goal}

User context:
{st.session_state.context}

Minimum acceptable outcome:
{st.session_state.minimum_outcome}

Desired outcome:
{st.session_state.desired_outcome}

Conversation transcript:
{conversation_text}
"""

    return call_ai(system_prompt, user_prompt, temperature=0.5)


def create_pdf_report():
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()
    story = []

    def add_title(text):
        story.append(Paragraph(text, styles["Title"]))
        story.append(Spacer(1, 12))

    def add_heading(text):
        story.append(Paragraph(text, styles["Heading2"]))
        story.append(Spacer(1, 8))

    def add_text(text):
        safe_text = str(text).replace("\n", "<br/>")
        story.append(Paragraph(safe_text, styles["BodyText"]))
        story.append(Spacer(1, 8))

    add_title("Optimum Negotiation Report")

    add_text(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    add_text(f"User: {st.session_state.user_name or 'Anonymous'}")

    add_heading("Scenario")
    add_text(st.session_state.scenario)

    add_heading("Goal")
    add_text(st.session_state.goal)

    add_heading("Context")
    add_text(st.session_state.context)

    add_heading("Minimum Acceptable Outcome")
    add_text(st.session_state.minimum_outcome)

    add_heading("Desired Outcome")
    add_text(st.session_state.desired_outcome)

    if st.session_state.scorecard:
        add_heading("Scorecard")
        scorecard = st.session_state.scorecard

        for key, value in scorecard.items():
            if isinstance(value, list):
                add_text(f"<b>{key.replace('_', ' ').title()}:</b><br/>" + "<br/>".join([f"- {item}" for item in value]))
            else:
                add_text(f"<b>{key.replace('_', ' ').title()}:</b> {value}")

    if st.session_state.improved_answers:
        add_heading("Improved Answers")
        add_text(st.session_state.improved_answers)

    if st.session_state.final_script:
        add_heading("Personalized Negotiation Script")
        add_text(st.session_state.final_script)

    add_heading("Conversation Transcript")
    add_text(get_conversation_text())

    doc.build(story)

    buffer.seek(0)
    return buffer


def reset_app():
    keys_to_reset = [
        "messages",
        "simulation_started",
        "simulation_finished",
        "scorecard",
        "improved_answers",
        "final_script",
        "scenario",
        "goal",
        "context",
        "minimum_outcome",
        "desired_outcome",
        "opponent_personality",
        "difficulty",
        "user_name",
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    init_session_state()
    st.rerun()


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.title("🎯 Optimum")
    st.caption("AI Negotiation Coach")

    st.markdown("---")

    st.subheader("MVP Settings")

    st.session_state.user_name = st.text_input(
        "Your name",
        value=st.session_state.user_name,
        placeholder="Example: Rachid"
    )

    st.session_state.scenario = st.selectbox(
        "Negotiation scenario",
        [
            "Salary raise negotiation",
            "Job offer negotiation",
            "Freelance price negotiation",
            "Client discount negotiation",
            "Used car purchase negotiation",
            "Rent reduction negotiation",
            "Business partnership negotiation",
            "Custom scenario"
        ],
        index=0 if not st.session_state.scenario else [
            "Salary raise negotiation",
            "Job offer negotiation",
            "Freelance price negotiation",
            "Client discount negotiation",
            "Used car purchase negotiation",
            "Rent reduction negotiation",
            "Business partnership negotiation",
            "Custom scenario"
        ].index(st.session_state.scenario)
        if st.session_state.scenario in [
            "Salary raise negotiation",
            "Job offer negotiation",
            "Freelance price negotiation",
            "Client discount negotiation",
            "Used car purchase negotiation",
            "Rent reduction negotiation",
            "Business partnership negotiation",
            "Custom scenario"
        ]
        else 0
    )

    if st.session_state.scenario == "Custom scenario":
        custom_scenario = st.text_input("Describe your custom scenario")
        if custom_scenario:
            st.session_state.scenario = custom_scenario

    st.session_state.opponent_personality = st.selectbox(
        "Opponent personality",
        [
            "Friendly but cautious",
            "Tough and skeptical",
            "Price-focused",
            "Dismissive and impatient",
            "Emotional and defensive",
            "Analytical and data-driven"
        ],
        index=0
    )

    st.session_state.difficulty = st.selectbox(
        "Difficulty",
        ["Easy", "Medium", "Hard"],
        index=1
    )

    st.markdown("---")

    if st.button("Reset simulation", use_container_width=True):
        reset_app()

    st.markdown("---")

    st.markdown(
        """
        <div class="small-note">
        Monetization idea:<br>
        Free: 1 simulation<br>
        Paid: €49 full report + 5 practice rounds<br>
        Premium: €99 salary negotiation pack
        </div>
        """,
        unsafe_allow_html=True
    )

    stripe_link = "https://buy.stripe.com/your-payment-link"

    st.link_button(
        "Upgrade placeholder",
        stripe_link,
        use_container_width=True
    )


# =========================
# HEADER
# =========================

st.markdown('<div class="main-title">Optimum</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Practice the conversation before money is on the table.</div>',
    unsafe_allow_html=True
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        """
        <div class="metric-card">
        <b>1. Simulate</b><br>
        Practice with a realistic negotiation opponent.
        </div>
        """,
        unsafe_allow_html=True
    )

with col_b:
    st.markdown(
        """
        <div class="metric-card">
        <b>2. Score</b><br>
        Get feedback on confidence, anchoring and objection handling.
        </div>
        """,
        unsafe_allow_html=True
    )

with col_c:
    st.markdown(
        """
        <div class="metric-card">
        <b>3. Improve</b><br>
        Download a personal negotiation script.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")


# =========================
# SETUP FORM
# =========================

if not st.session_state.simulation_started:
    st.subheader("Set up your negotiation")

    with st.form("setup_form"):
        st.session_state.goal = st.text_area(
            "What do you want to achieve?",
            placeholder="Example: I want to negotiate a €7,000 salary increase.",
            height=90
        )

        st.session_state.context = st.text_area(
            "Describe your situation",
            placeholder="Example: I have been in the role for 2 years, delivered strong results, and recently took on more responsibilities.",
            height=120
        )

        col1, col2 = st.columns(2)

        with col1:
            st.session_state.minimum_outcome = st.text_input(
                "Minimum acceptable outcome",
                placeholder="Example: €3,000 increase or a clear promotion plan"
            )

        with col2:
            st.session_state.desired_outcome = st.text_input(
                "Desired outcome",
                placeholder="Example: €7,000 increase"
            )

        start = st.form_submit_button("Start negotiation simulation", use_container_width=True)

        if start:
            if not st.session_state.goal or not st.session_state.context:
                st.warning("Please enter at least your goal and situation.")
            else:
                st.session_state.simulation_started = True

                opening_prompt = f"""
Start the negotiation simulation.

Scenario:
{st.session_state.scenario}

User goal:
{st.session_state.goal}

User context:
{st.session_state.context}

You are the opponent. Start with a realistic opening line.
"""

                opening_reply = call_ai(
                    get_roleplay_system_prompt(),
                    opening_prompt,
                    temperature=0.8
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": opening_reply
                    }
                )

                st.rerun()


# =========================
# ROLEPLAY CHAT
# =========================

if st.session_state.simulation_started and not st.session_state.simulation_finished:
    st.subheader("Negotiation simulation")

    st.info(
        "You are now in the negotiation. Respond naturally. When you are done, click 'Finish and analyze'."
    )

    for message in st.session_state.messages:
        with st.chat_message("assistant" if message["role"] == "assistant" else "user"):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your negotiation response...")

    if user_input:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        with st.spinner("Opponent is responding..."):
            reply = get_ai_opponent_reply()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply
            }
        )

        st.rerun()

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Finish and analyze", use_container_width=True):
            if len(st.session_state.messages) < 3:
                st.warning("Have at least one exchange before analyzing.")
            else:
                with st.spinner("Generating your negotiation scorecard..."):
                    st.session_state.scorecard = generate_scorecard()

                with st.spinner("Finding stronger answers..."):
                    st.session_state.improved_answers = generate_improved_answers()

                with st.spinner("Creating your personalized script..."):
                    st.session_state.final_script = generate_final_script()

                st.session_state.simulation_finished = True
                st.rerun()

    with col2:
        if st.button("Restart", use_container_width=True):
            reset_app()


# =========================
# RESULTS
# =========================

if st.session_state.simulation_finished:
    st.subheader("Your negotiation analysis")

    scorecard = st.session_state.scorecard

    if scorecard:
        st.markdown("### Scorecard")

        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric("Overall", scorecard.get("overall_score", 0))

        with metric_cols[1]:
            st.metric("Confidence", scorecard.get("confidence_score", 0))

        with metric_cols[2]:
            st.metric("Anchoring", scorecard.get("anchoring_score", 0))

        with metric_cols[3]:
            st.metric("Closing", scorecard.get("closing_score", 0))

        metric_cols_2 = st.columns(4)

        with metric_cols_2[0]:
            st.metric("Clarity", scorecard.get("clarity_score", 0))

        with metric_cols_2[1]:
            st.metric("Objection handling", scorecard.get("objection_handling_score", 0))

        with metric_cols_2[2]:
            st.metric("Concession control", scorecard.get("concession_control_score", 0))

        with metric_cols_2[3]:
            st.metric("Empathy", scorecard.get("empathy_score", 0))

        st.markdown("### Summary")
        st.write(scorecard.get("summary", ""))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Strengths")
            for item in scorecard.get("strengths", []):
                st.write(f"- {item}")

        with col2:
            st.markdown("#### Weaknesses")
            for item in scorecard.get("weaknesses", []):
                st.write(f"- {item}")

        with col3:
            st.markdown("#### Missed opportunities")
            for item in scorecard.get("missed_opportunities", []):
                st.write(f"- {item}")

        st.markdown("#### Next training focus")
        st.info(scorecard.get("next_training_focus", ""))

    if st.session_state.improved_answers:
        st.markdown("---")
        st.markdown("### Stronger answers you could have used")
        st.markdown(st.session_state.improved_answers)

    if st.session_state.final_script:
        st.markdown("---")
        st.markdown("### Personalized negotiation script")
        st.markdown(st.session_state.final_script)

    st.markdown("---")

    st.markdown("### Conversation transcript")

    for message in st.session_state.messages:
        role = "You" if message["role"] == "user" else "Opponent"
        st.markdown(f"**{role}:** {message['content']}")

    st.markdown("---")

    pdf_buffer = create_pdf_report()

    st.download_button(
        label="Download PDF report",
        data=pdf_buffer,
        file_name="optimum_negotiation_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run another simulation", use_container_width=True):
            reset_app()

    with col2:
        st.link_button(
            "Upgrade to full practice pack",
            "https://buy.stripe.com/your-payment-link",
            use_container_width=True
        )


# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    """
    <div class="cta-box">
    <h3>Optimum</h3>
    <p>
    Practice salary, client, business, and price negotiations before the real conversation happens.
    </p>
    <p>
    MVP positioning: Free simulation, paid report, premium preparation pack.
    </p>
    </div>
    """,
    unsafe_allow_html=True
)