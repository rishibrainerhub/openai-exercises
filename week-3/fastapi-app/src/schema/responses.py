from pydantic import BaseModel


class APIResponse(BaseModel):
    result: str
