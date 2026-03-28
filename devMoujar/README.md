uvicorn app.main:app 

ollama run llama3.2:1b 

streamlit run app/frontend.py


docker run -d --name deepsearch-neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/deepsearch neo4j:5



docker run -it -e POSTGRES_USER=root  -e POSTGRES_DB=demo -e POSTGRES_PASSWORD=toor -p 5432:5432 -v "./dev/tmp/mount:/var/lib/postgresql/data"  --name database postgres
