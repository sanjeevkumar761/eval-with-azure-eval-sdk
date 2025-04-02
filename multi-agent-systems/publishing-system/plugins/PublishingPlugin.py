import json
import os
import logging

from typing import Annotated
from semantic_kernel.functions import kernel_function
import asyncio
import json
import os
from .MCPClient import MCPClient

import logging
logging.basicConfig(level=logging.WARNING)  # Default logging level
logger = logging.getLogger("agent_collaboration")
logger.setLevel(logging.INFO)  # Application-specific logging level

class PublishingPlugin:
    """ 
    The class for PublishingPlugin.

    """


    @kernel_function(
        name="get_text_words_count",
        description="Gets word count",)
    def get_text_words_count(self,
                                          text: Annotated[str,"text"]) -> Annotated[str, "word count"]:

       logger.info(f"Text received in et_text_words_count: {text}")
       word_count = len(text.split())
       return word_count
    
    @kernel_function(
        name="get_letters_count",
        description="Gets letter count",)
    def get_letters_count(self,
                                          text: Annotated[str,"text"]) -> Annotated[str, "letters count"]:

       logger.info(f"Text received in get_letters_count: {text}")
       letters_count = len(text.replace(" ", "").replace("\n", ""))
       return letters_count
    
    @kernel_function(
        name="get_poem",
        description="Gets poem",)
    
    async def get_poem(self,
                                          topic: Annotated[str,"topic"]) -> Annotated[str, "poem"]:

        logger.info(f"Topic received in get_poem: {topic}")
        client = MCPClient()
        content = ""
        try:
            await client.connect_to_sse_server(server_url="http://localhost:8080/sse")
            result = await client.session.call_tool("get_poem", {"topic": topic})
            content = result.content
            print(content)
        finally:
            await client.cleanup()
        return content
        
