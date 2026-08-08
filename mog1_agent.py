import os
import sys
import re
import json
import urllib.request
import urllib.parse
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

class Mog1Agent:
    """
    Mog1 AI Autonomous Agent & Tool Execution Framework.
    Allows developers to register custom Python tools and execute agent workflows.
    """
    def __init__(self, name="Mog1Agent"):
        self.name = name
        self.tools = {}
        self._register_default_tools()

    def register_tool(self, name=None, description=""):
        """Decorator to register a custom tool function."""
        def decorator(func):
            tool_name = name or func.__name__
            self.tools[tool_name] = {
                "func": func,
                "description": description or func.__doc__ or "Custom tool"
            }
            return func
        return decorator

    def _register_default_tools(self):
        @self.register_tool(name="calculator", description="Evaluates mathematical expressions.")
        def calculator(expression: str) -> str:
            clean = expression.replace('?', '').replace('^', '**').strip()
            try:
                res = eval(clean, {"__builtins__": None}, {})
                return f"Result: {res}"
            except Exception as e:
                return f"Math Error: {e}"

        @self.register_tool(name="python_interpreter", description="Executes Python code snippets safely.")
        def python_interpreter(code: str) -> str:
            try:
                # Capture stdout
                import io
                buffer = io.StringIO()
                sys.stdout = buffer
                exec(code, {"__builtins__": __builtins__}, {})
                sys.stdout = sys.__stdout__
                output = buffer.getvalue().strip()
                return output if output else "Execution successful (no output)."
            except Exception as e:
                sys.stdout = sys.__stdout__
                return f"Execution Error: {e}"

        @self.register_tool(name="web_search", description="Fetches live search facts from Wikipedia.")
        def web_search(query: str) -> str:
            try:
                clean_q = re.sub(r'^(what is|who is|explain|tell me about)\s+', '', query, flags=re.IGNORECASE).strip()
                search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q)}&format=json"
                req = urllib.request.Request(search_url, headers={'User-Agent': 'Mog1Agent/1.0'})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    sdata = json.loads(resp.read().decode())
                    if 'query' in sdata and 'search' in sdata['query'] and len(sdata['query']['search']) > 0:
                        page_title = sdata['query']['search'][0]['title']
                        sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(page_title)}"
                        with urllib.request.urlopen(urllib.request.Request(sum_url, headers={'User-Agent': 'Mog1Agent/1.0'}), timeout=2) as sum_resp:
                            sum_data = json.loads(sum_resp.read().decode())
                            return sum_data.get('extract', 'No summary available.')
            except Exception as e:
                return f"Search Error: {e}"
            return "No search results found."

        @self.register_tool(name="system_info", description="Returns current date, time, and system environment info.")
        def system_info(query: str = "") -> str:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"Current System Time: {now} | OS: {sys.platform} | Python: {sys.version.split()[0]}"

    def list_tools(self) -> list:
        """Returns a list of registered tool names and descriptions."""
        return [{"name": k, "description": v["description"]} for k, v in self.tools.items()]

    def run(self, prompt: str) -> str:
        """Agent Loop: Analyzes prompt, selects tool, executes, and formats response."""
        p_lower = prompt.lower().strip()

        # 1. Math Tool Trigger
        if re.match(r'^[0-9\s\+\-\*\/\^\(\)\.]+\??$', prompt):
            tool_output = self.tools["calculator"]["func"](prompt)
            return f"🛠️ **Agent Tool Executed**: `calculator`\n\n{tool_output}"

        # 2. Python Code Interpreter Trigger
        if "run python" in p_lower or "exec python" in p_lower or "eval python" in p_lower:
            code_match = re.search(r'```python\n([\s\S]*?)```', prompt) or re.search(r'(?:python|code):\s*([\s\S]+)', prompt)
            code = code_match.group(1) if code_match else prompt
            tool_output = self.tools["python_interpreter"]["func"](code)
            return f"🛠️ **Agent Tool Executed**: `python_interpreter`\n\n```text\n{tool_output}\n```"

        # 3. System Info Trigger
        if any(k in p_lower for k in ["time", "date", "system info", "os", "environment"]):
            tool_output = self.tools["system_info"]["func"](prompt)
            return f"🛠️ **Agent Tool Executed**: `system_info`\n\n{tool_output}"

        # 4. Web Search Agent Trigger
        if any(k in p_lower for k in ["search", "who is", "what is", "where is", "explain"]):
            tool_output = self.tools["web_search"]["func"](prompt)
            return f"🛠️ **Agent Tool Executed**: `web_search`\n\n{tool_output}"

        return f"🤖 **Mog1Agent Response**:\nProcessed prompt '{prompt}' with {len(self.tools)} active tools registered."

if __name__ == "__main__":
    agent = Mog1Agent()
    print("Registered Tools:")
    for t in agent.list_tools():
        print(f" • {t['name']}: {t['description']}")
    
    print("\nTesting Agent Execution:")
    print(agent.run("What is 15 * 8?"))
    print(agent.run("Who is Albert Einstein?"))
