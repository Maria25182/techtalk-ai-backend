"""
TechTalk AI - Backend
FastAPI server con integración a Groq API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from groq import Groq
import random

app = FastAPI(title="TechTalk AI API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class InterviewRequest(BaseModel):
    question_type: str

class FeedbackRequest(BaseModel):
    question: str
    user_response: str
    question_type: str

# Banco de preguntas
QUESTIONS = {
    "behavioral": [
        "Cuéntame sobre un proyecto difícil en el que trabajaste y cómo lo resolviste.",
        "Describe una situación donde tuviste que aprender una tecnología nueva rápidamente.",
        "¿Cuál ha sido tu mayor desafío técnico y cómo lo superaste?",
    ],
    "coding": [
        "Explícame cómo implementarías una función para encontrar duplicados en una lista.",
        "Describe tu enfoque para optimizar una consulta SQL lenta.",
        "Explica cómo harías un merge entre dos listas ordenadas.",
    ],
    "system_design": [
        "Diseña un sistema de URL shortener como bit.ly.",
        "¿Cómo diseñarías un pipeline de datos para procesar millones de registros diarios?",
        "Explica cómo arquitecturarías un sistema de notificaciones en tiempo real.",
    ]
}

@app.get("/")
async def root():
    return {"status": "ok", "message": "TechTalk AI API"}

@app.post("/api/get-question")
async def get_question(request: InterviewRequest):
    try:
        question_type = request.question_type
        if question_type not in QUESTIONS:
            raise HTTPException(status_code=400, detail="Invalid type")
        
        question = random.choice(QUESTIONS[question_type])
        return {"question": question, "type": question_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/get-feedback")
async def get_feedback(request: FeedbackRequest):
    try:
        system_prompt = f"""Eres un entrevistador técnico. Evalúa esta respuesta:

Pregunta: {request.question}
Respuesta: {request.user_response}

Proporciona feedback en este formato:

CLARIDAD: [1-5]

FORTALEZAS:
- [punto 1]
- [punto 2]

MEJORAS:
- [punto 1]
- [punto 2]

Sé específico y constructivo."""

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=400,
        )
        
        feedback_text = chat_completion.choices[0].message.content
        
        return {
            "clarity_score": 4,
            "feedback": feedback_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
