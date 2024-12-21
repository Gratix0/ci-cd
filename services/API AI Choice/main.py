from fastapi import FastAPI, Depends, HTTPException
from AI.tags_model import convert_prompt_to_tag
from AI.answer_model import convert_tag_to_answer
import uvicorn
from typing import List

app = FastAPI(
    title="API-AI-Choice"
)

async def user_questions_to_currect_tag(question: dict) -> List[str]:
    try:
        return await convert_prompt_to_tag(question["question"]["question"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prompt: {e}")


async def tag_to_AI_answer(currect_tags: str) -> str:
    try:
        return await convert_tag_to_answer(currect_tags.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

@app.post("/answer_to_bot")
async def answer_to_bot(tags=Depends(user_questions_to_currect_tag)):
    """
    Endpoint for processing user questions and returning AI-generated answers.
    """
    try:
        currect_tags_str = " ".join(tags)
        ans = await tag_to_AI_answer(currect_tags_str)
        return {"status": "200", "result": ans}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")



if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
