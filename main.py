"""
TechTalk AI - Backend (Bilingual Edition)
FastAPI + Groq API for AI-powered interview practice in Spanish & English
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import random
import json

app = FastAPI(title="TechTalk AI - Bilingual")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    language: str = "es"  # es o en

class FeedbackResponse(BaseModel):
    feedback: str
    clarity_score: int
    strengths: list[str]
    improvements: list[str]

# Banco de preguntas - ESPAÑOL
QUESTIONS_ES = {
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
        "Explica cómo implementarías un sistema de caché simple.",
    ],
    "system_design": [
        "¿Cómo diseñarías un sistema de autenticación de usuarios?",
        "Explica cómo diseñarías una API REST para un blog.",
        "¿Cómo estructurarías una base de datos para un sistema de e-commerce?",
        "Diseña un sistema simple de notificaciones en tiempo real.",
        "¿Cómo manejarías el escalamiento de una aplicación web?",
    ]
}

# Banco de preguntas - ENGLISH
QUESTIONS_EN = {
    "behavioral": [
        "Tell me about a technical project you're proud of.",
        "Describe a time when you had to learn a new technology quickly.",
        "Tell me about a difficult bug you solved.",
        "Tell me about a time you had to work with a tight deadline.",
        "Describe a situation where you had to collaborate with a difficult team.",
    ],
    "coding": [
        "Explain how you would solve the problem of finding duplicates in an array.",
        "How would you implement a function to validate if a string is a palindrome?",
        "Explain your approach to designing a binary search function.",
        "How would you optimize a slow SQL query?",
        "Explain how you would implement a simple cache system.",
    ],
    "system_design": [
        "How would you design a user authentication system?",
        "Explain how you would design a REST API for a blog.",
        "How would you structure a database for an e-commerce system?",
        "Design a simple real-time notification system.",
        "How would you handle scaling a web application?",
    ]
}

@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "message": "TechTalk AI Backend (Bilingual)"}

@app.get("/api/question/{question_type}")
async def get_question(question_type: str, language: str = "es"):
    """
    Obtiene una pregunta random según el tipo e idioma
    """
    # Validar tipo de pregunta
    if question_type not in ["behavioral", "coding", "system_design"]:
        raise HTTPException(status_code=400, detail="Invalid question type")
    
    # Validar idioma
    if language not in ["es", "en"]:
        raise HTTPException(status_code=400, detail="Invalid language")
    
    # Seleccionar banco de preguntas
    questions_bank = QUESTIONS_ES if language == "es" else QUESTIONS_EN
    
    # Obtener pregunta aleatoria
    question = random.choice(questions_bank[question_type])
    
    return {
        "question": question,
        "type": question_type,
        "language": language
    }

@app.post("/api/feedback", response_model=FeedbackResponse)
async def generate_feedback(request: FeedbackRequest):
    """
    Genera feedback sobre la respuesta del usuario usando Groq
    Feedback en el idioma especificado
    """
    try:
        # Prompt según idioma
        if request.language == "es":
            prompt = f"""Eres un experto en entrevistas técnicas. 

Un candidato respondió a una pregunta de tipo {request.question_type} con la siguiente respuesta:

"{request.user_response}"

Analiza la respuesta y proporciona feedback constructivo en ESPAÑOL en formato JSON con esta estructura exacta:

{{
  "clarity_score": [número del 1-5, donde 5 es excelente],
  "strengths": ["punto fuerte 1", "punto fuerte 2"],
  "improvements": ["área a mejorar 1", "área a mejorar 2", "área a mejorar 3"],
  "feedback": "Un párrafo breve (2-3 oraciones) con feedback general en español"
}}

Sé constructivo, específico y amable. El candidato está practicando.
Responde SOLO con el JSON, sin texto adicional."""
        else:
            prompt = f"""You are a technical interview expert.

A candidate answered a {request.question_type} question with this response:

"{request.user_response}"

Analyze the response and provide constructive feedback in ENGLISH in JSON format with this exact structure:

{{
  "clarity_score": [number from 1-5, where 5 is excellent],
  "strengths": ["strength 1", "strength 2"],
  "improvements": ["improvement area 1", "improvement area 2", "improvement area 3"],
  "feedback": "A brief paragraph (2-3 sentences) with general feedback in English"
}}

Be constructive, specific, and kind. The candidate is practicing.
Respond ONLY with the JSON, no additional text."""

        # Llamada a Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a technical interview expert providing constructive feedback."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1000,
        )

        # Extraer respuesta
        response_text = chat_completion.choices[0].message.content
        
        # Parsear JSON
        feedback_data = json.loads(response_text)
        
        return FeedbackResponse(
            feedback=feedback_data.get("feedback", ""),
            clarity_score=feedback_data.get("clarity_score", 3),
            strengths=feedback_data.get("strengths", []),
            improvements=feedback_data.get("improvements", [])
        )
        
    except json.JSONDecodeError:
        # Fallback según idioma
        if request.language == "es":
            return FeedbackResponse(
                feedback="Gracias por tu respuesta. Intenta ser más específico y dar ejemplos concretos.",
                clarity_score=3,
                strengths=["Respondiste la pregunta", "Mostraste interés"],
                improvements=[
                    "Agrega más detalles específicos",
                    "Menciona métricas o resultados concretos",
                    "Estructura tu respuesta mejor"
                ]
            )
        else:
            return FeedbackResponse(
                feedback="Thanks for your response. Try to be more specific and give concrete examples.",
                clarity_score=3,
                strengths=["You answered the question", "You showed interest"],
                improvements=[
                    "Add more specific details",
                    "Mention concrete metrics or results",
                    "Structure your answer better"
                ]
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating feedback: {str(e)}"
        )

@app.get("/api/stats")
async def get_stats():
    """Estadísticas"""
    return {
        "total_questions_es": sum(len(q) for q in QUESTIONS_ES.values()),
        "total_questions_en": sum(len(q) for q in QUESTIONS_EN.values()),
        "question_types": list(QUESTIONS_ES.keys()),
        "languages": ["es", "en"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
