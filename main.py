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

# Banco de preguntas - ESPAÑOL (20 por categoría)
QUESTIONS_ES = {
    "behavioral": [
        # Proyectos y logros
        "Cuéntame sobre un proyecto técnico del que estés orgulloso/a.",
        "Describe el proyecto más desafiante en el que hayas trabajado.",
        "Háblame de una vez que tuviste que liderar un proyecto técnico.",
        "Cuéntame sobre un proyecto donde tuviste que trabajar con tecnologías nuevas.",
        "Describe una funcionalidad compleja que hayas implementado.",
        # Aprendizaje y adaptación
        "Describe una vez que tuviste que aprender una nueva tecnología rápidamente.",
        "Cuéntame sobre una vez que cometiste un error técnico. ¿Qué aprendiste?",
        "Háblame de una situación donde recibiste feedback negativo. ¿Cómo lo manejaste?",
        "Describe cómo te mantienes actualizado con nuevas tecnologías.",
        # Resolución de problemas
        "Háblame de un bug difícil que hayas resuelto.",
        "Cuéntame sobre una vez que optimizaste el rendimiento de un sistema.",
        "Describe una situación donde tuviste que debugging en producción.",
        "Háblame de un problema técnico que parecía imposible de resolver.",
        # Trabajo en equipo
        "Cuéntame sobre una vez que tuviste que trabajar con un deadline ajustado.",
        "Describe una situación donde tuviste que colaborar con un equipo difícil.",
        "Háblame de un desacuerdo técnico que tuviste con un colega. ¿Cómo lo resolviste?",
        "Cuéntame sobre una vez que tuviste que explicar un concepto técnico a alguien no técnico.",
        "Describe cómo manejas el code review de otros desarrolladores.",
        # Iniciativa y impacto
        "Háblame de una vez que mejoraste un proceso en tu equipo.",
        "Cuéntame sobre una contribución que hiciste más allá de tus responsabilidades.",
    ],
    "coding": [
        # Estructuras de datos básicas
        "Explícame cómo resolverías el problema de encontrar duplicados en un array.",
        "¿Cómo implementarías una función para validar si una cadena es un palíndromo?",
        "Describe cómo invertirías un array sin usar métodos built-in.",
        "Explica cómo encontrarías el elemento más frecuente en un array.",
        "¿Cómo removerías elementos duplicados de un array manteniendo el orden?",
        # Algoritmos de búsqueda y ordenamiento
        "Explica tu approach para diseñar una función de búsqueda binaria.",
        "¿Cómo implementarías un algoritmo de ordenamiento? Explica tu elección.",
        "Describe cómo buscarías un elemento en una matriz 2D ordenada.",
        # Strings y manipulación
        "¿Cómo validarías si dos strings son anagramas?",
        "Explica cómo comprimirías un string (ej: 'aabbbcccc' → 'a2b3c4').",
        "¿Cómo encontrarías la subcadena más larga sin caracteres repetidos?",
        # Lógica y problemas matemáticos
        "Explica cómo determinarías si un número es primo.",
        "¿Cómo calcularías el factorial de un número? Compara iterativo vs recursivo.",
        "Describe cómo generarías la secuencia de Fibonacci.",
        # Bases de datos y optimización
        "¿Cómo optimizarías una query SQL que está tardando mucho?",
        "Explica cuándo usarías un índice en una base de datos.",
        "Describe la diferencia entre INNER JOIN y LEFT JOIN con un ejemplo.",
        # Diseño de funciones
        "Explica cómo implementarías un sistema de caché simple.",
        "¿Cómo diseñarías una función de rate limiting (límite de requests)?",
        "Describe cómo implementarías un sistema de retry con backoff exponencial.",
    ],
    "system_design": [
        # Autenticación y seguridad
        "¿Cómo diseñarías un sistema de autenticación de usuarios?",
        "Explica cómo implementarías autenticación con JWT tokens.",
        "Describe cómo manejarías la autorización y permisos en una aplicación.",
        "¿Cómo diseñarías un sistema de reset de contraseñas seguro?",
        # APIs y microservicios
        "Explica cómo diseñarías una API REST para un blog.",
        "Describe la diferencia entre REST y GraphQL. ¿Cuándo usarías cada uno?",
        "¿Cómo estructurarías una arquitectura de microservicios?",
        "Explica cómo manejarías versionado de APIs.",
        # Bases de datos
        "¿Cómo estructurarías una base de datos para un sistema de e-commerce?",
        "Explica cuándo elegirías SQL vs NoSQL.",
        "Describe cómo diseñarías el schema de una red social básica.",
        "¿Cómo manejarías transacciones en una base de datos distribuida?",
        # Sistemas en tiempo real
        "Diseña un sistema simple de notificaciones en tiempo real.",
        "¿Cómo implementarías un chat en vivo?",
        "Explica cómo diseñarías un sistema de tracking en tiempo real (como Uber).",
        # Escalabilidad y performance
        "¿Cómo manejarías el escalamiento de una aplicación web?",
        "Explica qué es caching y dónde lo implementarías.",
        "Describe cómo diseñarías un sistema para manejar millones de usuarios.",
        "¿Cómo implementarías un sistema de cola de mensajes (message queue)?",
        # Monitoreo y confiabilidad
        "Explica cómo diseñarías un sistema de logging y monitoreo.",
        "¿Cómo manejarías el rollback de un deployment con problemas?",
    ]
}

# Banco de preguntas - ENGLISH (20 por categoría)
QUESTIONS_EN = {
    "behavioral": [
        # Projects and achievements
        "Tell me about a technical project you're proud of.",
        "Describe the most challenging project you've worked on.",
        "Tell me about a time you had to lead a technical project.",
        "Tell me about a project where you worked with new technologies.",
        "Describe a complex feature you implemented.",
        # Learning and adaptation
        "Describe a time when you had to learn a new technology quickly.",
        "Tell me about a time you made a technical mistake. What did you learn?",
        "Tell me about a situation where you received negative feedback. How did you handle it?",
        "Describe how you stay updated with new technologies.",
        # Problem solving
        "Tell me about a difficult bug you solved.",
        "Tell me about a time you optimized a system's performance.",
        "Describe a situation where you had to debug in production.",
        "Tell me about a technical problem that seemed impossible to solve.",
        # Teamwork
        "Tell me about a time you had to work with a tight deadline.",
        "Describe a situation where you had to collaborate with a difficult team.",
        "Tell me about a technical disagreement with a colleague. How did you resolve it?",
        "Tell me about a time you had to explain a technical concept to a non-technical person.",
        "Describe how you handle code reviews from other developers.",
        # Initiative and impact
        "Tell me about a time you improved a process in your team.",
        "Tell me about a contribution you made beyond your responsibilities.",
    ],
    "coding": [
        # Basic data structures
        "Explain how you would solve the problem of finding duplicates in an array.",
        "How would you implement a function to validate if a string is a palindrome?",
        "Describe how you would reverse an array without using built-in methods.",
        "Explain how you would find the most frequent element in an array.",
        "How would you remove duplicate elements from an array while maintaining order?",
        # Search and sorting algorithms
        "Explain your approach to designing a binary search function.",
        "How would you implement a sorting algorithm? Explain your choice.",
        "Describe how you would search for an element in a sorted 2D matrix.",
        # Strings and manipulation
        "How would you validate if two strings are anagrams?",
        "Explain how you would compress a string (e.g., 'aabbbcccc' → 'a2b3c4').",
        "How would you find the longest substring without repeating characters?",
        # Logic and mathematical problems
        "Explain how you would determine if a number is prime.",
        "How would you calculate the factorial of a number? Compare iterative vs recursive.",
        "Describe how you would generate the Fibonacci sequence.",
        # Databases and optimization
        "How would you optimize a slow SQL query?",
        "Explain when you would use an index in a database.",
        "Describe the difference between INNER JOIN and LEFT JOIN with an example.",
        # Function design
        "Explain how you would implement a simple cache system.",
        "How would you design a rate limiting function?",
        "Describe how you would implement a retry system with exponential backoff.",
    ],
    "system_design": [
        # Authentication and security
        "How would you design a user authentication system?",
        "Explain how you would implement authentication with JWT tokens.",
        "Describe how you would handle authorization and permissions in an application.",
        "How would you design a secure password reset system?",
        # APIs and microservices
        "Explain how you would design a REST API for a blog.",
        "Describe the difference between REST and GraphQL. When would you use each?",
        "How would you structure a microservices architecture?",
        "Explain how you would handle API versioning.",
        # Databases
        "How would you structure a database for an e-commerce system?",
        "Explain when you would choose SQL vs NoSQL.",
        "Describe how you would design the schema for a basic social network.",
        "How would you handle transactions in a distributed database?",
        # Real-time systems
        "Design a simple real-time notification system.",
        "How would you implement a live chat system?",
        "Explain how you would design a real-time tracking system (like Uber).",
        # Scalability and performance
        "How would you handle scaling a web application?",
        "Explain what caching is and where you would implement it.",
        "Describe how you would design a system to handle millions of users.",
        "How would you implement a message queue system?",
        # Monitoring and reliability
        "Explain how you would design a logging and monitoring system.",
        "How would you handle rolling back a problematic deployment?",
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
