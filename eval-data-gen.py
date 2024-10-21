from azure.ai.evaluation.simulator import Simulator

import asyncio
from simulator import Simulator
from azure.identity import DefaultAzureCredential
import wikipedia
import os
from typing import List, Dict, Any, Optional
# Prepare the text to send to the simulator
wiki_search_term = "Leonardo da vinci"
wiki_title = wikipedia.search(wiki_search_term)[0]
wiki_page = wikipedia.page(wiki_title)
text = wiki_page.summary[:5000]

