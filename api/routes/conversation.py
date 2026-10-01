from fastapi import APIRouter, Response, status

from api.dependencies import agent

router = APIRouter()


@router.delete("/conversation", status_code=status.HTTP_204_NO_CONTENT)
async def clear_conversation() -> Response:
    await agent.clear_conversation()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
