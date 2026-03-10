import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from llm import LLM
from knowno.pipeline import EntityExtractor, TaskHandler, ResponseGenerator
from knowno.embedding import EnvironmentMatcher, EmbeddingSelector
from knowno.classify import AmbiguityClassifier
from memory.session_store import SessionStore

DATASET_PATH = Path(__file__).parent / "ambik_dataset" / "ambik_test_900.csv"
GREETING = "hey, robot kitchen"
FAREWELL = "thank you, robot kitchen"


@st.cache_resource
def load_pipeline():
    llm = LLM("ollama:llama3.1:8b", {"max_new_tokens": 150, "temperature": 0.7})
    embed_model = LLM("ollama:mxbai-embed-large:latest", {})
    extractor = EntityExtractor(llm)
    env_matcher = EnvironmentMatcher(DATASET_PATH)
    embedding_selector = EmbeddingSelector(embed_model)
    classifier = AmbiguityClassifier(llm)
    responder = ResponseGenerator(llm)
    handler = TaskHandler(extractor, env_matcher, embedding_selector, classifier)
    return handler, responder


def init_state():
    defaults = {
        "stage": "idle",       # idle → awaiting_task → chatting → ended
        "session_id": None,
        "task": None,
        "messages": [],
        "session_store": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_message(role, content, meta=None):
    st.session_state.messages.append({"role": role, "content": content, "meta": meta})


def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                with st.expander("Details"):
                    st.json(msg["meta"])


def handle_input(user_input, handler, responder):
    stage = st.session_state.stage
    add_message("user", user_input)

    if stage == "idle":
        if user_input.strip().lower() == GREETING:
            st.session_state.session_id = st.session_state.session_store.new_session_id()
            st.session_state.stage = "awaiting_task"
            add_message("assistant", f"Hello! I'm ready. What task would you like me to do?\n\n`Session: {st.session_state.session_id}`")
        else:
            add_message("assistant", f'Please say **"{GREETING}"** to begin.')

    elif stage == "awaiting_task":
        task = user_input.strip()
        handler.start_task(task)
        st.session_state.task = task
        st.session_state.session_store.save_session({
            "session_id": st.session_state.session_id,
            "task": task,
            "environment": handler.environment,
            "history": [],
        })
        st.session_state.stage = "chatting"
        add_message("assistant", f"Got it! Task: **{task}**\nNow tell me the steps. Say *\"{FAREWELL}\"* when done.")

    elif stage == "chatting":
        if user_input.strip().lower() == FAREWELL:
            st.session_state.stage = "ended"
            add_message("assistant", "You're welcome! Session ended. Goodbye! 👋")
            return

        result = handler.handle_step(user_input)
        entities = result.get("entities", {})
        actions = entities.get("actions", []) if isinstance(entities, dict) else []
        cls = result.get("classification")

        robot_response = ""
        meta = {
            "actions": actions,
            "objects": entities.get("objects", []) if isinstance(entities, dict) else [],
            "top_objects": [(obj, round(score, 4)) for obj, score in result.get("top_objects", [])],
        }

        if cls:
            action_str = ", ".join(actions)
            robot_response = responder.generate(user_input, cls, action_str)
            meta["classification"] = cls.get("status")
            meta["label"] = cls.get("label")
            meta["viable_objects"] = cls.get("viable_objects", [])

        st.session_state.session_store.add_turn(
            st.session_state.session_id,
            st.session_state.task,
            handler.environment,
            user_input,
            robot_response,
        )
        add_message("assistant", robot_response or "Done. What's next?", meta)


def main():
    st.set_page_config(page_title="Robot Kitchen", page_icon="🤖", layout="centered")
    st.title("Robot Kitchen")

    init_state()

    if st.session_state.session_store is None:
        st.session_state.session_store = SessionStore()

    handler, responder = load_pipeline()

    if st.session_state.stage == "idle":
        st.info(f'Say **"{GREETING}"** to start.')

    render_messages()

    if st.session_state.stage != "ended":
        placeholder = {
            "idle": f'"{GREETING}"',
            "awaiting_task": "Enter task (e.g. prepare breakfast)...",
            "chatting": "Enter step query...",
        }.get(st.session_state.stage, "")

        if user_input := st.chat_input(placeholder):
            handle_input(user_input, handler, responder)
            st.rerun()
    else:
        if st.button("Start new session"):
            for key in ["stage", "session_id", "task", "messages", "session_store"]:
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
