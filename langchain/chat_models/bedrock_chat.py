from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.llms.bedrock import Bedrock
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from pydantic import Field

class ChatBedrock(BaseChatModel, Bedrock):
    
    llm: Bedrock = Field(default=None, exclude=False)
    
    r"""Wrapper around Bedrock's APIs for large language model.

    Example:
        .. code-block:: python

            from langchain.llms import Bedrock
            model = ChatBedrock(model_id="<model_name>", llm=<Bedrock>)
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        if not self.llm:
            raise NameError("Plese ensure the Bedrock LLM is loaded")
        return self.llm._llm_type

    @property
    def lc_serializable(self) -> bool:
        return True

    def _convert_one_message_to_text(self, message: BaseMessage) -> str:
        return f"{message.content}"

    def _convert_messages_to_text(self, messages: List[BaseMessage]) -> str:
        """Format a list of strings into a single string with necessary newlines.

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary newlines.
        """
        return "".join(
            self._convert_one_message_to_text(message) for message in messages
        )

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Format a list of messages into a full prompt for the model

        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.

        Returns:
            str: Combined string with necessary HUMAN_PROMPT and AI_PROMPT tags.
        """
        messages = messages.copy()  # don't mutate the original list

        if not isinstance(messages[-1], AIMessage):
            messages.append(AIMessage(content=""))
        text = self._convert_messages_to_text(messages)
        return (
            text.rstrip()
        )  # trim off the trailing ' ' that might come from the "Assistant: "

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        response = self.llm._call(prompt=prompt, **kwargs)
        message = AIMessage(content=response)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages=messages, stop=stop, run_manager=run_manager, kwargs=kwargs)

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        if not self.count_tokens:
            raise NameError("Plese ensure the Bedrock LLM is loaded")
        return self.count_tokens(text)