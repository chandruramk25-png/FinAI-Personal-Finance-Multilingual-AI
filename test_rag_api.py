# test_rag_api.py
import urllib.request
import json

queries = [
    ('en', 'What is the 50/30/20 budget rule and how does it apply to me?'),
    ('hi', '50/30/20 बजट नियम क्या है और इसे कैसे लागू करें?'),
    ('ta', '50/30/20 பட்ஜெட் விதி என்றால் என்ன?'),
    ('en', 'What are the best tax deductions under Section 80C?'),
    ('hi', 'मुझे कितने महीने का इमरजेंसी फंड रखना चाहिए?')
]

with open("test_rag_output.txt", "w", encoding="utf-8") as out_f:
    for lang, q in queries:
        req = urllib.request.Request(
            'http://127.0.0.1:5000/api/chatbot',
            data=json.dumps({'message': q, 'language': lang}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            out_f.write(f"=== LANG: {lang} | QUERY: {q} ===\n")
            out_f.write(f"RAG Active: {data.get('rag_active')}\n")
            sources = data.get('rag_sources', [])
            out_f.write(f"RAG Sources Count: {len(sources)}\n")
            for i, src in enumerate(sources):
                out_f.write(f"  Source [{i+1}]: {src.get('title')} ({src.get('category')}) | Score: {src.get('score')}\n")
            out_f.write(f"Response:\n{data.get('response')}\n")
            out_f.write("-" * 60 + "\n\n")

print("RAG Test completed. Output written to test_rag_output.txt")
