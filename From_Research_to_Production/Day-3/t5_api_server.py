from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import uvicorn

app = FastAPI()

# Завантаження моделі та токенізатора
print("Завантаження моделі та токенізатора...")
model_path = "local_t5_model"  # Шлях до локально збереженої моделі
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Використовуємо локальну директорію замість 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_path)
model.to(device)
model.eval()

print(f"Модель завантажено на пристрій: {device}")


class Question(BaseModel):
    text: str


def generate_answer(question: str, max_length: int = 256, num_beams: int = 4, temperature: float = 0.7):
    input_text = f"question: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


@app.post("/ask")
async def ask_question(question: Question):
    try:
        answer = generate_answer(question.text)
        return {"question": question.text, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "T5 API is running. Use /ask endpoint to get answers."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

print("Сервер запущено. Використовуйте /ask для отримання відповідей.")
