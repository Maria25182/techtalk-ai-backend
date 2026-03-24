"""
TechTalk AI - Backend (Complete Edition)
FastAPI + Groq API
Incluye: Behavioral, Coding, System Design, Data Engineering, Machine Learning, SQL
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import random
import json

app = FastAPI(title="TechTalk AI - Complete Edition")

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
    language: str = "es"

class FeedbackResponse(BaseModel):
    feedback: str
    clarity_score: int
    strengths: list[str]
    improvements: list[str]

# ============================================
# BANCO DE PREGUNTAS - ESPAÑOL
# ============================================

QUESTIONS_ES = {
    "behavioral": [
        "Cuéntame sobre un proyecto técnico del que estés orgulloso/a.",
        "Describe el proyecto más desafiante en el que hayas trabajado.",
        "Háblame de una vez que tuviste que liderar un proyecto técnico.",
        "Describe una vez que tuviste que aprender una nueva tecnología rápidamente.",
        "Cuéntame sobre una vez que cometiste un error técnico. ¿Qué aprendiste?",
        "Háblame de un bug difícil que hayas resuelto.",
        "Cuéntame sobre una vez que optimizaste el rendimiento de un sistema.",
        "Cuéntame sobre una vez que tuviste que trabajar con un deadline ajustado.",
        "Describe una situación donde tuviste que colaborar con un equipo difícil.",
        "Háblame de un desacuerdo técnico que tuviste con un colega. ¿Cómo lo resolviste?",
        "Cuéntame sobre una vez que tuviste que explicar un concepto técnico a alguien no técnico.",
        "Háblame de una vez que mejoraste un proceso en tu equipo.",
        "Cuéntame sobre una contribución que hiciste más allá de tus responsabilidades.",
        "Describe cómo manejas el code review de otros desarrolladores.",
        "Háblame de una situación donde recibiste feedback negativo. ¿Cómo lo manejaste?",
    ],
    
    "coding": [
        "Explícame cómo resolverías el problema de encontrar duplicados en un array.",
        "¿Cómo implementarías una función para validar si una cadena es un palíndromo?",
        "Describe cómo invertirías un array sin usar métodos built-in.",
        "Explica tu approach para diseñar una función de búsqueda binaria.",
        "¿Cómo implementarías un algoritmo de ordenamiento? Explica tu elección.",
        "¿Cómo validarías si dos strings son anagramas?",
        "Explica cómo comprimirías un string (ej: 'aabbbcccc' → 'a2b3c4').",
        "Explica cómo determinarías si un número es primo.",
        "¿Cómo calcularías el factorial de un número? Compara iterativo vs recursivo.",
        "Describe cómo generarías la secuencia de Fibonacci.",
        "Explica cómo implementarías un sistema de caché simple.",
        "¿Cómo diseñarías una función de rate limiting?",
        "Explica cómo encontrarías el elemento más frecuente en un array.",
        "¿Cómo removerías elementos duplicados de un array manteniendo el orden?",
        "¿Cómo encontrarías la subcadena más larga sin caracteres repetidos?",
    ],
    
    "system_design": [
        "¿Cómo diseñarías un sistema de autenticación de usuarios?",
        "Explica cómo implementarías autenticación con JWT tokens.",
        "Explica cómo diseñarías una API REST para un blog.",
        "Describe la diferencia entre REST y GraphQL. ¿Cuándo usarías cada uno?",
        "¿Cómo estructurarías una arquitectura de microservicios?",
        "Explica cómo manejarías versionado de APIs.",
        "¿Cómo estructurarías una base de datos para un sistema de e-commerce?",
        "Diseña un sistema simple de notificaciones en tiempo real.",
        "¿Cómo implementarías un chat en vivo?",
        "¿Cómo manejarías el escalamiento de una aplicación web?",
        "Explica qué es caching y dónde lo implementarías.",
        "Describe cómo diseñarías un sistema para manejar millones de usuarios.",
        "¿Cómo implementarías un sistema de cola de mensajes (message queue)?",
        "Explica cómo diseñarías un sistema de logging y monitoreo.",
        "¿Cómo manejarías el rollback de un deployment con problemas?",
    ],
    
    "data_engineering": [
        # ETL y Pipelines
        "Explica qué es un pipeline ETL y sus componentes principales.",
        "¿Cómo diseñarías un pipeline de datos para procesar millones de registros diarios?",
        "Describe la diferencia entre ETL y ELT. ¿Cuándo usarías cada uno?",
        "¿Cómo manejarías datos duplicados en un pipeline de datos?",
        "Explica qué es idempotencia en data pipelines y por qué es importante.",
        # Big Data y Spark
        "¿Qué es Apache Spark y cuándo lo usarías sobre pandas?",
        "Explica la diferencia entre transformaciones y acciones en Spark.",
        "¿Cómo optimizarías un job de Spark que está corriendo lento?",
        "Describe qué es data partitioning y por qué es importante.",
        "Explica qué son transformaciones lazy en Spark.",
        # Data Warehousing
        "¿Cuál es la diferencia entre un Data Lake y un Data Warehouse?",
        "Explica qué es una arquitectura medallion (bronze, silver, gold).",
        "¿Cómo diseñarías un modelo dimensional (star schema)?",
        "Describe qué es slowly changing dimension (SCD) tipo 2.",
        # Data Quality
        "¿Cómo garantizarías la calidad de datos en un pipeline?",
        "Explica qué es schema evolution y cómo lo manejarías.",
        "¿Cómo detectarías y manejarías datos late-arriving?",
        # Herramientas
        "Compara Apache Airflow vs otras herramientas de orquestación.",
        "Explica cuándo usarías batch processing vs stream processing.",
        "¿Qué es Delta Lake y qué problemas resuelve?",
    ],
    
    "machine_learning": [
        # Fundamentos
        "Explica la diferencia entre supervised y unsupervised learning.",
        "¿Qué es overfitting y cómo lo prevendrías?",
        "Describe qué es la diferencia entre clasificación y regresión.",
        "Explica qué es cross-validation y por qué es importante.",
        "¿Qué métricas usarías para evaluar un modelo de clasificación?",
        # Feature Engineering
        "¿Cómo manejarías valores faltantes (missing values) en un dataset?",
        "Explica qué es feature scaling y cuándo es necesario.",
        "Describe técnicas para manejar datos desbalanceados.",
        "¿Cómo detectarías y manejarías outliers?",
        # Modelos
        "Explica cómo funciona un árbol de decisión.",
        "¿Cuál es la diferencia entre Random Forest y Gradient Boosting?",
        "Describe cuándo usarías regresión lineal vs regresión logística.",
        "Explica qué es regularización (L1 vs L2).",
        # MLOps y Deployment
        "¿Cómo deployarías un modelo de ML a producción?",
        "Explica qué es model drift y cómo lo detectarías.",
        "¿Cómo monitorearías el performance de un modelo en producción?",
        "Describe qué es A/B testing en el contexto de ML.",
        # Deep Learning básico
        "Explica qué es una red neuronal en términos simples.",
        "¿Cuándo considerarías usar deep learning vs modelos tradicionales?",
        "Describe qué es transfer learning.",
    ],
    
    "sql_databases": [
        # SQL Básico
        "¿Cómo optimizarías una query SQL que está tardando mucho?",
        "Explica la diferencia entre INNER JOIN y LEFT JOIN con un ejemplo.",
        "¿Cuándo usarías una subquery vs un JOIN?",
        "Describe qué hace la cláusula HAVING vs WHERE.",
        # Optimización
        "Explica cuándo usarías un índice en una base de datos.",
        "¿Qué es un índice compuesto y cuándo lo usarías?",
        "Describe qué es query execution plan y cómo lo interpretas.",
        "¿Cómo identificarías queries lentas en una base de datos?",
        # Diseño
        "Explica qué es normalización de bases de datos.",
        "¿Cuándo denormalizarías una tabla y por qué?",
        "Describe la diferencia entre OLTP y OLAP.",
        # Transacciones
        "Explica qué son las propiedades ACID.",
        "¿Qué es un deadlock y cómo lo prevendrías?",
        "Describe qué son los niveles de aislamiento en transacciones.",
        # Window Functions
        "Explica qué son window functions y da un ejemplo.",
        "¿Cuál es la diferencia entre ROW_NUMBER, RANK y DENSE_RANK?",
        # Performance
        "¿Cómo optimizarías una tabla con millones de registros?",
        "Explica qué es partitioning en bases de datos.",
        "Describe estrategias para manejar queries con múltiples JOINs.",
    ]
}

# ============================================
# BANCO DE PREGUNTAS - ENGLISH
# ============================================

QUESTIONS_EN = {
    "behavioral": [
        "Tell me about a technical project you're proud of.",
        "Describe the most challenging project you've worked on.",
        "Tell me about a time you had to lead a technical project.",
        "Describe a time when you had to learn a new technology quickly.",
        "Tell me about a time you made a technical mistake. What did you learn?",
        "Tell me about a difficult bug you solved.",
        "Tell me about a time you optimized a system's performance.",
        "Tell me about a time you had to work with a tight deadline.",
        "Describe a situation where you had to collaborate with a difficult team.",
        "Tell me about a technical disagreement with a colleague. How did you resolve it?",
        "Tell me about a time you had to explain a technical concept to a non-technical person.",
        "Tell me about a time you improved a process in your team.",
        "Tell me about a contribution you made beyond your responsibilities.",
        "Describe how you handle code reviews from other developers.",
        "Tell me about a situation where you received negative feedback. How did you handle it?",
    ],
    
    "coding": [
        "Explain how you would solve the problem of finding duplicates in an array.",
        "How would you implement a function to validate if a string is a palindrome?",
        "Describe how you would reverse an array without using built-in methods.",
        "Explain your approach to designing a binary search function.",
        "How would you implement a sorting algorithm? Explain your choice.",
        "How would you validate if two strings are anagrams?",
        "Explain how you would compress a string (e.g., 'aabbbcccc' → 'a2b3c4').",
        "Explain how you would determine if a number is prime.",
        "How would you calculate the factorial of a number? Compare iterative vs recursive.",
        "Describe how you would generate the Fibonacci sequence.",
        "Explain how you would implement a simple cache system.",
        "How would you design a rate limiting function?",
        "Explain how you would find the most frequent element in an array.",
        "How would you remove duplicate elements from an array while maintaining order?",
        "How would you find the longest substring without repeating characters?",
    ],
    
    "system_design": [
        "How would you design a user authentication system?",
        "Explain how you would implement authentication with JWT tokens.",
        "Explain how you would design a REST API for a blog.",
        "Describe the difference between REST and GraphQL. When would you use each?",
        "How would you structure a microservices architecture?",
        "Explain how you would handle API versioning.",
        "How would you structure a database for an e-commerce system?",
        "Design a simple real-time notification system.",
        "How would you implement a live chat system?",
        "How would you handle scaling a web application?",
        "Explain what caching is and where you would implement it.",
        "Describe how you would design a system to handle millions of users.",
        "How would you implement a message queue system?",
        "Explain how you would design a logging and monitoring system.",
        "How would you handle rolling back a problematic deployment?",
    ],
    
    "data_engineering": [
        # ETL and Pipelines
        "Explain what an ETL pipeline is and its main components.",
        "How would you design a data pipeline to process millions of records daily?",
        "Describe the difference between ETL and ELT. When would you use each?",
        "How would you handle duplicate data in a data pipeline?",
        "Explain what idempotency means in data pipelines and why it's important.",
        # Big Data and Spark
        "What is Apache Spark and when would you use it over pandas?",
        "Explain the difference between transformations and actions in Spark.",
        "How would you optimize a slow Spark job?",
        "Describe what data partitioning is and why it's important.",
        "Explain what lazy transformations are in Spark.",
        # Data Warehousing
        "What's the difference between a Data Lake and a Data Warehouse?",
        "Explain what a medallion architecture (bronze, silver, gold) is.",
        "How would you design a dimensional model (star schema)?",
        "Describe what a slowly changing dimension (SCD) type 2 is.",
        # Data Quality
        "How would you ensure data quality in a pipeline?",
        "Explain what schema evolution is and how you would handle it.",
        "How would you detect and handle late-arriving data?",
        # Tools
        "Compare Apache Airflow vs other orchestration tools.",
        "Explain when you would use batch processing vs stream processing.",
        "What is Delta Lake and what problems does it solve?",
    ],
    
    "machine_learning": [
        # Fundamentals
        "Explain the difference between supervised and unsupervised learning.",
        "What is overfitting and how would you prevent it?",
        "Describe the difference between classification and regression.",
        "Explain what cross-validation is and why it's important.",
        "What metrics would you use to evaluate a classification model?",
        # Feature Engineering
        "How would you handle missing values in a dataset?",
        "Explain what feature scaling is and when it's necessary.",
        "Describe techniques for handling imbalanced data.",
        "How would you detect and handle outliers?",
        # Models
        "Explain how a decision tree works.",
        "What's the difference between Random Forest and Gradient Boosting?",
        "Describe when you would use linear regression vs logistic regression.",
        "Explain what regularization is (L1 vs L2).",
        # MLOps and Deployment
        "How would you deploy an ML model to production?",
        "Explain what model drift is and how you would detect it.",
        "How would you monitor a model's performance in production?",
        "Describe what A/B testing means in the context of ML.",
        # Basic Deep Learning
        "Explain what a neural network is in simple terms.",
        "When would you consider using deep learning vs traditional models?",
        "Describe what transfer learning is.",
    ],
    
    "sql_databases": [
        # SQL Basics
        "How would you optimize a slow SQL query?",
        "Explain the difference between INNER JOIN and LEFT JOIN with an example.",
        "When would you use a subquery vs a JOIN?",
        "Describe what the HAVING clause does vs WHERE.",
        # Optimization
        "Explain when you would use an index in a database.",
        "What is a composite index and when would you use it?",
        "Describe what a query execution plan is and how to interpret it.",
        "How would you identify slow queries in a database?",
        # Design
        "Explain what database normalization is.",
        "When would you denormalize a table and why?",
        "Describe the difference between OLTP and OLAP.",
        # Transactions
        "Explain what ACID properties are.",
        "What is a deadlock and how would you prevent it?",
        "Describe what transaction isolation levels are.",
        # Window Functions
        "Explain what window functions are and give an example.",
        "What's the difference between ROW_NUMBER, RANK, and DENSE_RANK?",
        # Performance
        "How would you optimize a table with millions of records?",
        "Explain what partitioning is in databases.",
        "Describe strategies for handling queries with multiple JOINs.",
    ]
}

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "message": "TechTalk AI Backend - Complete Edition",
        "categories": list(QUESTIONS_ES.keys())
    }

@app.get("/api/question/{question_type}")
async def get_question(question_type: str, language: str = "es"):
    """Obtiene una pregunta random según el tipo e idioma"""
    # Validar tipo de pregunta
    valid_types = ["behavioral", "coding", "system_design", "data_engineering", "machine_learning", "sql_databases"]
    if question_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid question type. Valid types: {valid_types}")
    
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
    """Genera feedback sobre la respuesta del usuario usando Groq"""
    try:
        # Prompt según idioma
        if request.language == "es":
            prompt = f"""Eres un experto en entrevistas técnicas especialmente en {request.question_type}. 

Un candidato respondió con:
"{request.user_response}"

Proporciona feedback constructivo en ESPAÑOL en formato JSON:
{{
  "clarity_score": [1-5],
  "strengths": ["punto fuerte 1", "punto fuerte 2"],
  "improvements": ["mejora 1", "mejora 2", "mejora 3"],
  "feedback": "Feedback general en 2-3 oraciones"
}}

Sé específico, constructivo y amable. Responde SOLO con JSON."""
        else:
            prompt = f"""You are a technical interview expert specializing in {request.question_type}.

A candidate answered:
"{request.user_response}"

Provide constructive feedback in ENGLISH in JSON format:
{{
  "clarity_score": [1-5],
  "strengths": ["strength 1", "strength 2"],
  "improvements": ["improvement 1", "improvement 2", "improvement 3"],
  "feedback": "General feedback in 2-3 sentences"
}}

Be specific, constructive, and kind. Respond ONLY with JSON."""

        # Llamada a Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a technical interview expert."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1000,
        )

        response_text = chat_completion.choices[0].message.content
        feedback_data = json.loads(response_text)
        
        return FeedbackResponse(
            feedback=feedback_data.get("feedback", ""),
            clarity_score=feedback_data.get("clarity_score", 3),
            strengths=feedback_data.get("strengths", []),
            improvements=feedback_data.get("improvements", [])
        )
        
    except json.JSONDecodeError:
        # Fallback
        if request.language == "es":
            return FeedbackResponse(
                feedback="Gracias por tu respuesta. Intenta ser más específico.",
                clarity_score=3,
                strengths=["Respondiste la pregunta", "Mostraste interés"],
                improvements=["Agrega más detalles", "Menciona ejemplos concretos", "Estructura mejor"]
            )
        else:
            return FeedbackResponse(
                feedback="Thanks for your response. Try to be more specific.",
                clarity_score=3,
                strengths=["You answered the question", "You showed interest"],
                improvements=["Add more details", "Mention concrete examples", "Better structure"]
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Estadísticas"""
    return {
        "categories": list(QUESTIONS_ES.keys()),
        "total_questions_es": sum(len(q) for q in QUESTIONS_ES.values()),
        "total_questions_en": sum(len(q) for q in QUESTIONS_EN.values()),
        "questions_per_category": {
            cat: {"es": len(QUESTIONS_ES[cat]), "en": len(QUESTIONS_EN[cat])}
            for cat in QUESTIONS_ES.keys()
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
