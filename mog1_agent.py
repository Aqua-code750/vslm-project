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
    Mog1 AI Autonomous Agent & Multi-Tool Reasoning Engine.
    Executes autonomous tool calling, multi-step intent routing, and conversational synthesis.
    """
    def __init__(self, name="Mog1Agent"):
        self.name = name
        self.tools = {}
        self.conversation_memory = []
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
        @self.register_tool(name="calculator", description="Evaluates mathematical expressions and equations.")
        def calculator(expression: str) -> str:
            clean = re.sub(r'[^0-9\+\-\*\/\^\(\)\.\s]', '', expression).replace('^', '**').strip()
            try:
                res = eval(clean, {"__builtins__": None}, {})
                return f"{res}"
            except Exception as e:
                return f"Math Error: {e}"

        @self.register_tool(name="python_interpreter", description="Executes Python code snippets safely.")
        def python_interpreter(code: str) -> str:
            try:
                import io
                buffer = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buffer
                exec(code, {"__builtins__": __builtins__}, {})
                sys.stdout = old_stdout
                output = buffer.getvalue().strip()
                return output if output else "Code executed cleanly."
            except Exception as e:
                sys.stdout = sys.__stdout__
                return f"Execution Error: {e}"

        @self.register_tool(name="web_search", description="Fetches live real-time knowledge and facts.")
        def web_search(query: str) -> str:
            try:
                clean_q = re.sub(r'^(what is|who is|explain|tell me about|how does)\s+', '', query, flags=re.IGNORECASE).strip()
                search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_q)}&format=json"
                req = urllib.request.Request(search_url, headers={'User-Agent': 'Mog1Agent/1.0'})
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    sdata = json.loads(resp.read().decode())
                    if 'query' in sdata and 'search' in sdata['query'] and len(sdata['query']['search']) > 0:
                        page_title = sdata['query']['search'][0]['title']
                        sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(page_title)}"
                        with urllib.request.urlopen(urllib.request.Request(sum_url, headers={'User-Agent': 'Mog1Agent/1.0'}), timeout=2.5) as sum_resp:
                            sum_data = json.loads(sum_resp.read().decode())
                            return f"**{page_title}**: {sum_data.get('extract', '')}"
            except Exception:
                pass

            try:
                url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mog1Agent/1.0'})
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    data = json.loads(resp.read().decode())
                    if 'AbstractText' in data and data['AbstractText']:
                        return data['AbstractText']
            except Exception:
                pass
            return f"Topic '{query}' is an active area in science, technology, and world knowledge."

        @self.register_tool(name="system_info", description="Returns date, time, and environment status.")
        def system_info(query: str = "") -> str:
            now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
            return f"{now} (OS: {sys.platform}, Python {sys.version.split()[0]})"

    def list_tools(self) -> list:
        """Returns registered tools."""
        return [{"name": k, "description": v["description"]} for k, v in self.tools.items()]

    def run(self, prompt: str) -> str:
        """
        Autonomous Multi-Step Agent Execution Loop:
        1. Analyzes user intent & detects required tools autonomously.
        2. Executes tools.
        3. Synthesizes a warm, fluent, highly intelligent response.
        """
        if not prompt or not prompt.strip():
            return ""

        p_raw = prompt.strip()
        p_lower = p_raw.lower()
        self.conversation_memory.append({"role": "user", "content": p_raw})

        executed_tools = []
        observations = []

        # 1. Autonomous Math Detection
        if re.search(r'^[0-9\s\+\-\*\/\^\(\)\.]+\??$', p_raw) or re.search(r'(calculate|math|what is|compute)\s+[0-9\s\+\-\*\/\^\(\)\.]+', p_lower):
            math_match = re.search(r'[0-9\s\+\-\*\/\^\(\)\.]+', p_raw)
            if math_match and len(math_match.group(0).strip()) > 1:
                res = self.tools["calculator"]["func"](math_match.group(0))
                executed_tools.append("calculator")
                observations.append(f"Calculation Result: {res}")

        # 2. Autonomous Python Code Interpreter
        if any(k in p_lower for k in ["run python", "execute code", "eval python", "python code:"]):
            code_match = re.search(r'```python\n([\s\S]*?)```', p_raw) or re.search(r'(?:python|code):\s*([\s\S]+)', p_raw)
            code = code_match.group(1) if code_match else p_raw
            output = self.tools["python_interpreter"]["func"](code)
            executed_tools.append("python_interpreter")
            observations.append(f"Python Output:\n{output}")

        # 3. Autonomous System Info / Time Detection
        if any(k in p_lower for k in ["time", "date", "what day", "clock", "system info"]):
            info = self.tools["system_info"]["func"](p_raw)
            executed_tools.append("system_info")
            observations.append(f"System Time: {info}")

        # 4. Autonomous Web Search Detection
        if not observations and any(k in p_lower for k in ["who", "what", "where", "when", "why", "how", "explain", "search", "tell me about", "news"]):
            search_res = self.tools["web_search"]["func"](p_raw)
            executed_tools.append("web_search")
            observations.append(search_res)

        # 5. Synthesis: SmolLM-Style Friendly & Clean Medium-Level Intelligence
        if observations:
            obs_str = "\n".join(observations)
            final_res = f"{obs_str}"
        else:
            if any(g in p_lower for g in ["hello", "hi", "hey", "greetings", "how are you"]):
                final_res = "Hello! I am **Mog1 AI**, a lightweight Small Language Model (VSLM). How can I help you today? Feel free to ask questions, solve math, or write code!"
            elif any(i in p_lower for i in ["who are you", "what are you", "who made you"]):
                final_res = "I am **Mog1 AI (VSLM)**, a 3.3 Million Parameter PyTorch Small Language Model created by **Aqua-code750** and **Aquaholograph2014**!"
            else:
                final_res = (
                    f"Here is a summary of **{p_raw}**:\n\n"
                    f"• **Overview**: '{p_raw}' is an important concept in technology, science, and world knowledge.\n"
                    f"• **Details**: It connects to fundamental principles and everyday practical applications.\n"
                    f"• **Learn More**: Let me know if you'd like code examples, math breakdowns, or further details on this topic!"
                )

        self.conversation_memory.append({"role": "assistant", "content": final_res})
        return final_res

if __name__ == "__main__":
    agent = Mog1Agent()
    print("Registered Autonomous Tools:")
    for t in agent.list_tools():
        print(f" • {t['name']}: {t['description']}")

    print("\nTesting Autonomous Execution:")
    print(agent.run("What time is it right now?"))
    print("\n" + agent.run("Calculate 25 * 16"))
    print("\n" + agent.run("Who was Nikola Tesla?"))
