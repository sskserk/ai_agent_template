import logging

log = logging.getLogger(__name__)
from .state import AgentState

class AgentStateWrapper:
    def __init__(self, model_with_tools=None):
        self.model_with_tools = model_with_tools

    def assistant(self, state: AgentState) -> AgentState:
        print(f"Invoking model with {state.model_dump_json()} messages")
        new_messages = self.model_with_tools.invoke(state.messages)

        messages_to_append = []

        if isinstance(new_messages, list):
            log.info(f"Received {len(new_messages)} messages from the assistant.")

            messages_to_append = new_messages
        else:
            log.info(f"Received single message from the assistant {new_messages}")
            messages_to_append = [new_messages]
            state.messages.append(new_messages)


        for message in messages_to_append:
            state.messages.append(message)

        log.info(f"Total messages in state: {len(state.messages)}")

        return state