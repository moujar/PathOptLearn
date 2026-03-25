# PathOptLearn
The project employs LLMs to guide students through a series of videos on a specific topic. The video selection follows a branching path formula, where the next video is based on the student’s performance on a set of questions provided by an LLM.
Workflow:
1. Page: Video Selection (show YouTube embed)
2. Process: Whisper extracts transcript
3. Process: Mistral generates questions
4. Page: Quiz interface (MCQs)
5. Process: Mistral evaluates answers
6. Process: Calculate score + update database
7. Process: Rule-based next video recommendation
8. Page: Show recommendation + progress Dashboard

# The files
## API
The 'APIs.ipynb' file is the API file that we used to access the question generation and evaluation models. The models are hosted on Colab for now. The needed endpoints exist in this file.

## Pages
There are 3 pages:
- Log in: this one is self explainatory.
- Dashboard: this page shows the courses the user is currently following.
- Learning: this is the page of the course being currently followed.

# Mailer
This is the email verification process thru code sent to the user's email.

# Databases
The db.py is the users and their related information database file. The adaptive.db file is the current course info (the course being currently followed).

# As For The Recommendation System...

The recommender system  has the following workflow:
1. LLM (LLama) searches Youtube.
2. Finds a bunch of videos.
3. Looks for any video that has captions. 
4. If it finds one, it analyzes it is transcripts.
5. If not , it goes for the meta data.
6. Then the LLM picks the best fit

# To Do:

- Integrating a more advanced recommendation system.
- Assembling everything together.
- Having better pages' designs

# Running The App
- Run the APIs.ipynb
- Open Anaconda locally.
- Run app.py after downloading required packages.


Models directories links to download:
https://drive.google.com/drive/folders/15047-E2jRbeoHLkxsvlStT7cD20-UJwo?usp=sharing

https://drive.google.com/drive/folders/1dF2Zq3i4ki-X4tmkOt4txfhd1Y9SWHjL?usp=sharing
