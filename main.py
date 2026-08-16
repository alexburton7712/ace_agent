import asyncio
import os

from agent import Agent


async def main():
    agent = Agent()
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
        os._exit(0)