import ollama

if __name__ == "__main__":
    client = ollama.Client(host='http://130.61.18.190:11434')
    response = client.chat(model="llama3", messages=[{"role": "user", "content": "Hallo"}])
    print(response)

# docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
# docker exec -it ollama ollama run llama3