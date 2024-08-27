import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_qa(num_questions):
    questions = []
    qa_pairs = []

    for i in range(num_questions):
        # Generate question
        question_prompt = f"Create a question about DevOps. Question #{i+1}"
        question_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": question_prompt}]
        )
        question = question_response.choices[0].message.content.strip()
        questions.append(question)

        # Generate answer
        answer_prompt = f"Provide a brief answer (up to 10 sentences) to the following DevOps question: {question}"
        answer_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": answer_prompt}]
        )
        answer = answer_response.choices[0].message.content.strip()
        qa_pairs.append(f"Question: {question}\n\nAnswer: {answer}\n\n")

    return questions, qa_pairs

def save_to_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))

if __name__ == "__main__":
    num_questions = 5
    questions, qa_pairs = generate_qa(num_questions)

    save_to_file('devops_questions.txt', questions)
    save_to_file('devops_qa.txt', qa_pairs)

    print("Files successfully created: devops_questions.txt and devops_qa.txt")