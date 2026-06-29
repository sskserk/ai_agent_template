import asyncio
import logging
from pathlib import Path
from typing import Any
from appl.agents.smith.agent import AgentSmith
from appl.agents.smith.user_notifications import UserNotification
import datetime as date
import json
import uuid
from msg_publisher import MessagePublisher

log = logging.getLogger(__name__)


def setup_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file =  "app.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )

    log.info("Logging initialized. Output file: %s", log_file)


async def main_iteration(message_content: str, 
                         thread_id: str,
                         mesg_publisher: MessagePublisher
                         ):


    # await print_message_to_user(
    #     UserNotification(message="Final response:==============", type="final_response")
    # )
    #for user_message in user_messages:
    content = message_content

    math_agent = AgentSmith(thread_id=thread_id, message_printer=mesg_publisher.publish_message)
    log.info("AgentSmith initialized")



    await mesg_publisher.publish_message(
        UserNotification(message=f"{content}", 
                        type="user", 
                        message_id="user_" + uuid.uuid4().hex[:8]
                        )
    )
    # await print_message_to_user(
    #     UserNotification(message="[assistant] ", type="final_response", end="")
    # )
    
    await asyncio.sleep(1)
    is_first_stream_chunk = True
    pending_chunk = None
    
    first_chunk_message_id = None

    def extract_response_id(chunk: Any) -> str | None:
        response_metadata = getattr(chunk, "response_metadata", None)
        if isinstance(response_metadata, dict):
            response_id = response_metadata.get("id")
            if isinstance(response_id, str) and response_id.startswith("resp_"):
                return response_id

        chunk_id = getattr(chunk, "id", None)
        if isinstance(chunk_id, str) and chunk_id.startswith("resp_"):
            return chunk_id

        return None

    async for event in math_agent.astream_events(content):
        #log.debug("Received event: %s", json.dumps(event, indent=2, default=str))
        
        if event["event"] == "on_chat_model_stream":
            log.debug(event)
            
            chunk = event["data"]["chunk"]
            #if first_chunk_message_id is None:
            chunk_message_id = extract_response_id(chunk)
            if chunk_message_id is not None:
                log.info("First chunk message ID: %s", chunk_message_id)
                first_chunk_message_id = chunk_message_id
                is_first_stream_chunk = True
                log.info("First chunk: %s", first_chunk_message_id)            
            
            if chunk.content:

                    #first_chunk_message_id = chunk.id if hasattr(chunk, "id") else None
                
                if pending_chunk is not None and len(pending_chunk) > 0:
                    
                    await mesg_publisher.publish_message(
                        UserNotification(
                            message_id=first_chunk_message_id,
                            message=pending_chunk,
                            type="streamed_chunk",
                            end="",
                            is_start=is_first_stream_chunk,
                            is_end=False,
                        )
                    )
                    is_first_stream_chunk = False

                pending_chunk = chunk.content

    if pending_chunk is not None:
        await mesg_publisher.publish_message(
            UserNotification(
                message_id=first_chunk_message_id,
                message=pending_chunk,
                type="streamed_chunk",
                end="",
                is_start=False,
                is_end=True,
            )
        )
        
    await mesg_publisher.publish_message(UserNotification(message="", type="final_response", message_id=first_chunk_message_id))


    all_messages = await math_agent.get_all_messages()
    log.info("All messages in the conversation:")
    for msg in all_messages:
        log.debug(msg.message_id + " | " + msg.role + " | " + msg.content)



async def main():
    log.info("Application started")
    thread_id = date.datetime.now().strftime("%Y%m%d_%H%M%S")

    user_messages = [
        {"content": "Calculate the 2+2*3."},
        {"content": "Now, add 1 to the result. What is the final answer?"},
    ]
    
    mesg_publisher = MessagePublisher()
    
    for user_message in user_messages:
        await main_iteration(user_message["content"], 
                             thread_id= thread_id,
                             mesg_publisher=mesg_publisher
                             )
        
    log.info("Application finished successfully")


if __name__ == "__main__":
    setup_logging()
    try:
        asyncio.run(main())
    except Exception:
        log.exception("Unhandled exception in main")
        raise
