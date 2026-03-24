"""
TechTalk AI - Backend
FastAPI + Groq API for AI-powered interview practice
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import random
import json

app = FastAPI(title="TechTalk AI")

# CORS para permitir llamadas desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Modelos de datos
class FeedbackRequest(BaseModel):
    user_response: str
    question_type: str = "behavioral"

class FeedbackResponse(BaseModel):
    feedback: str
    clarity_score: int
    strengths: list[str]
    improvements: list[str]

# Banco de preguntas por categoría
QUESTIONS = {
    "behavioral": [
        "Cuéntame sobre un proyecto técnico del que estés orgulloso/a.",
        "Describe una vez que tuviste que aprender una nueva tecnología rápidamente.",
        "Háblame de un bug difícil que hayas resuelto.",
        "Cuéntame sobre una vez que tuviste que trabajar con un deadline ajustado.",
        "Describe una situación donde tuviste que colaborar con un equipo difícil.",
    ],
    "coding": [
        "Explícame cómo resolverías el problema de encontrar duplicados en un array.",
        "¿Cómo implementarías una función para validar si una cadena es un palíndromo?",
        "Explica tu approach para diseñar una función de búsqueda binaria.",
        "¿Cómo optimizarías una query SQL que está tardando mucho?",
        "Explica cómo implementarías un cache simple.",
    ],
    "system_design": [
        "¿Cómo diseñarías un sistema de autenticación de usuarios?",
        "Explica cómo diseñarías una API REST para un blog.",
        "¿Cómo estructurarías una base de datos para un sistema de e-commerce?",
        "Diseña un sistema simple de notificaciones.",
        "¿Cómo manejarías el escalamiento de una aplicación web?",
    ]
}

@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "message": "TechTalk AI Backend is running"}

@app.get("/api/question/{question_type}")
async def get_question(question_type: str):
    """
    Obtiene una pregunta random según el tipo
    """
    if question_type not in QUESTIONS:
        raise HTTPException(status_code=400, detail="Invalid question type")
    
    question = random.choice(QUESTIONS[question_type])
    return {"question": question, "type": question_type}

@app.post("/api/feedback", response_model=FeedbackResponse)
async def generate_feedback(request: FeedbackRequest):
    """
    Genera feedback sobre la respuesta del usuario usando Groq
    """
    try:
        # Prompt para Groq
        prompt = f"""Eres un experto en entrevistas técnicas. 

Un candidato respondió a una pregunta de tipo {request.question_type} con la siguiente respuesta:

"{request.user_response}"

Analiza la respuesta y proporciona feedback constructivo en formato JSON con esta estructura exacta:

{{
  "clarity_score": [número del 1-5, donde 5 es excelente],
  "strengths": ["punto fuerte 1", "punto fuerte 2"],
  "improvements": ["área a mejorar 1", "área a mejorar 2", "área a mejorar 3"],
  "feedback": "Un párrafo breve (2-3 oraciones) con feedback general"
}}

Sé constructivo, específico y amable. El candidato está practicando y aprendiendo.
Responde SOLO con el JSON, sin texto adicional."""

        # Llamada a Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en entrevistas técnicas que da feedback constructivo y específico."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",  # Modelo rápido y gratis de Groq
            temperature=0.7,
            max_tokens=1000,
        )

        # Extraer respuesta
        response_text = chat_completion.choices[0].message.content
        
        # Parsear JSON (Groq debería devolver JSON limpio)
        feedback_data = json.loads(response_text)
        
        return FeedbackResponse(
            feedback=feedback_data.get("feedback", ""),
            clarity_score=feedback_data.get("clarity_score", 3),
            strengths=feedback_data.get("strengths", []),
            improvements=feedback_data.get("improvements", [])
        )
        
    except json.JSONDecodeError:
        # Fallback si Groq no devuelve JSON válido
        return FeedbackResponse(
            feedback="Gracias por tu respuesta. Intenta ser más específico y dar ejemplos concretos.",
            clarity_score=3,
            strengths=["Respondiste la pregunta", "Mostraste interés"],
            improvements=[
                "Agrega más detalles específicos",
                "Menciona métricas o resultados concretos",
                "Estructura tu respuesta con inicio, desarrollo y conclusión"
            ]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating feedback: {str(e)}"
        )

@app.get("/api/stats")
async def get_stats():
    """
    Estadísticas simples (para futuro)
    """
    return {
        "total_questions": sum(len(q) for q in QUESTIONS.values()),
        "question_types": list(QUESTIONS.keys())
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
