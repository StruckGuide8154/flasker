# custom_tools.py
import requests
from bs4 import BeautifulSoup
import json
import os

class CustomTools:
    def __init__(self):
        self.load_tools()
    
    def load_tools(self):
        try:
            with open('custom_tools.json', 'r') as f:
                self.tools = json.load(f)
        except FileNotFoundError:
            self.tools = {
                "google": {
                    "command": "/google",
                    "url": "https://www.google.com/search?q=",
                    "description": "Search Google",
                    "enabled": True
                },
                "wiki": {
                    "command": "/wiki",
                    "url": "https://en.wikipedia.org/wiki/Special:Search?search=",
                    "description": "Search Wikipedia",
                    "enabled": True
                }
            }
            self.save_tools()
    
    def save_tools(self):
        with open('custom_tools.json', 'w') as f:
            json.dump(self.tools, f, indent=4)
    
    def execute_tool(self, command, query):
        tool = next((tool for tool in self.tools.values() if tool["command"] == command and tool["enabled"]), None)
        if not tool:
            return f"Tool not found or disabled: {command}"
        
        try:
            response = requests.get(tool["url"] + requests.utils.quote(query))
            response.raise_for_status()
            
            # Parse result (example using BeautifulSoup)
            soup = BeautifulSoup(response.text, 'html.parser')
            # This is a simple example - you'd want to customize the parsing based on the site
            result = soup.get_text()[:500] + "..."  # First 500 characters
            
            return f"Results from {tool['description']}:\n{result}"
        except Exception as e:
            return f"Error executing tool {command}: {str(e)}"

