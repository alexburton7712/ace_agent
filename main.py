import asyncio
import logging
import os

from agent import Agent

logging.basicConfig(
    level=os.getenv("ACE_LOG_LEVEL", "INFO").upper(),
    format=(
        "%(asctime)s %(levelname)s "
        "%(name)s: %(message)s"
    ),
)


async def main():
    logging.getLogger(__name__).info("Starting Ace")
    agent = Agent()
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
        os._exit(0)
