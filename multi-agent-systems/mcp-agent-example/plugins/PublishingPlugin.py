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
        name="get_contents_json",
        description="Gets content JSON",)
    def get_contents_json(self,
                                          text: Annotated[str,"text"]) -> Annotated[str, "JSON content"]:
        # Create a dictionary to hold the JSON data
        logger.info(f"Text received in get_contents_json: {text}")
        content_json = {
            "text": text
        }
        return json.dumps(content_json)  
    
    @kernel_function(
        name="validate_contents_json",
        description="Validates content JSON",)
    def validate_contents_json(self,
                                          text: Annotated[str,"json"]) -> Annotated[str, "validation result"]:
        # Create a dictionary to hold the JSON data
        logger.info(f"JSON received in validate_contents_json: {text}")
        try:
            # Attempt to parse the JSON
            parsed_json = json.loads(text)
            logger.info("JSON is valid.")
            return json.dumps({"status": "valid", "parsed_json": parsed_json})
        except json.JSONDecodeError as e:
            # Handle invalid JSON
            logger.error(f"Invalid JSON: {e}")
            return json.dumps({"status": "invalid", "error": str(e)})   
    
    @kernel_function(
        name="get_sci_poem",
        description="Gets scientific poem",)

    async def get_sci_poem(self,
                                          topic: Annotated[str,"topic"]) -> Annotated[str, "poem"]:

        logger.info(f"Topic received in get_sci_poem: {topic}")
        client = MCPClient()
        content = ""
        try:
            await client.connect_to_sse_server(server_url="http://localhost:8080/sse")
            result = await client.session.call_tool("get_sci_poem", {"topic": topic})
            content = result.content
            print(content)
        finally:
            await client.cleanup()
        return content
        
