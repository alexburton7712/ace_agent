from llm import generate_stream


class Agent:
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": self._load_system_prompt(),
            }
        ]

    def _load_system_prompt(self):
        with open(
            "assistant_prompt.md",
            "r",
            encoding="utf-8",
        ) as f:
            return f.read()

    async def run(self):
        while True:
            prompt = input("You: ")

            self.messages.append({
                "role": "user",
                "content": prompt,
            })

            await self.respond()

    async def respond(self):
        print("Ace:", end=" ", flush=True)

        response = ""
        showing_thinking = False
        showing_response = False

        async for chunk_type, chunk in generate_stream(
            self.messages
        ):
            if chunk_type == "thinking":
                if not showing_thinking:
                    print("<think>")
                    showing_thinking = True

                print(chunk, end="", flush=True)

            elif chunk_type == "response":
                if not showing_response:
                    if showing_thinking:
                        print("\n</think>")

                    showing_response = True

                print(chunk, end="", flush=True)
                response += chunk

        print()

        self.messages.append({
            "role": "assistant",
            "content": response,
        })