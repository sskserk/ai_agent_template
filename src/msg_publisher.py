
import datetime
from typing import Any
from appl.agents.smith.agent import UserNotification



class LexemPublisher:
    def __init__(self, notification_callback):
        self.notification_callback = notification_callback

        self.lexems = []
        self.prev_last_index = -1
        self.prev_message_id = None
        
        self.last_published_index = 0
        self.possible_formula_endings = [r"\)", r"\\)", r"\]", r"\\]"]
        self.possible_formula_starts = [r"\(", r"\\(", r"\[", r"\\["]

        self.formula_spotted = False
        self.formula_start_spotted = False


    async def publish_lexem(self, lexem: str, message_id: str) -> None:
        self.lexems.append(lexem)

        full_text = ''.join(self.lexems)

        # Check for the last occurrence of any possible formula ending
        last_index = -1
        for ending in self.possible_formula_endings:
            last_index = full_text.rfind(ending)
            if last_index > -1:
                self.formula_spotted = True
                break

        for start in self.possible_formula_starts:
            start_index = full_text.rfind(start)
            if start_index > -1 and (last_index == -1 or start_index > last_index):
                self.formula_start_spotted = True
                break

        is_break = '\n' in lexem or '\r' in lexem
        
        
        if self.prev_message_id is None:
            self.prev_message_id = message_id
           
#        log.debug(f"lexem [{lexem}], {is_break}, {self.formula_spotted}, {self.formula_start_spotted}, {last_index}, {self.formula_start_spotted}")

        if (self.prev_message_id != message_id) or \
            (last_index != -1 and last_index != self.prev_last_index) or \
            (is_break and not self.formula_spotted and not self.formula_start_spotted):
            self.prev_last_index = last_index
            await self.publish()
            
        self.prev_message_id = message_id

    async def publish(self) -> None:
        message = ''.join(self.lexems[self.last_published_index:])

        await self.notification_callback(message=message, message_id=self.prev_message_id)
        self.formula_spotted = False
        self.last_published_index = len(self.lexems)
        self.formula_start_spotted = False

    async def flush_unpublished(self) -> None:
        if self.last_published_index < len(self.lexems):
            await self.publish()

    async def get_content(self) -> str:
        return ''.join(self.lexems)


class MessagePublisher:
    def __init__(self):
        
        async def publish_callback(message: str, message_id: str) -> None:
            print(f"[lexem {message_id}] {message}", end="", flush=True)
    
        self._lexem_publisher = LexemPublisher(notification_callback=publish_callback)  # Default callback for lexem publishing
        #pass

    async def publish_message(self, notification: UserNotification):
        id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        def extract_text(payload: Any) -> str:
            if isinstance(payload, str):
                return payload

            if isinstance(payload, dict):
                return str(
                    payload.get("text")
                    or payload.get("output_text")
                    or payload.get("content")
                    or ""
                )

            if isinstance(payload, list):
                parts: list[str] = []
                for item in payload:
                    text = extract_text(item)
                    if text:
                        parts.append(text)
                return "".join(parts)

            return str(payload)

        if notification.type == "streamed_chunk":
            text = extract_text(notification.message)
            if text is None or len(text) == 0:
                return
            

                
            if notification.is_start:
                print(f"\n{id} [assistant / {notification.message_id}] {text}", end=notification.end, flush=True)
            if text:
                await self._lexem_publisher.publish_lexem(text, notification.message_id)
                #print(f"{text}", end=notification.end, flush=True)
                
            if notification.is_end:
                print(f"{id}  <= [assistant / {notification.message_id}] [end of stream]", end=notification.end, flush=True)
        elif notification.type == "tool_start":
            print(f"\n{id} [tool:start] {notification.message}", end=notification.end, flush=True)
        elif notification.type == "tool_end":
            print(f"\n{id} [tool:end] {notification.message}", end=notification.end, flush=True)
        elif notification.type == "user":
            print(f"\n{id} [user / {notification.message_id}] {notification.message}", end=notification.end, flush=True)
        elif notification.type == "final_response":
            #print(f"\n{id} [assistant / {notification.message_id}] Final response: {notification.message}", end=notification.end, flush=True)
            await self._lexem_publisher.flush_unpublished()
            pass
        else:
            pass
            
            #print(f"\n[Unknown notification {notification.type}] '{notification.message}' ", end=notification.end, flush=True)

        
        
