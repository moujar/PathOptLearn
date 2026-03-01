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

## Models
The models' training notebooks are:
1. For the evaluation model there are three files where we increased the training data in each time strating with 'adaptive_learning_finetune_pro.ipynb' to 'adaptive_learning_4datasets_FINAL.ipynb' to finally 'evaluation_model_FINAL_A100.ipynb', we have tested on 4 benchmarks, all of their files are uploaded here too.
2. The questions generation model's trainig notebook is 'question_generation_training.ipynb', and we tested it in the 'question_generation_inference.ipynb' notebook.

## The Main Interface
The 'youtube_transcript_app_3.py' file is the streamlit app that accesses the two models after scraping the transcript of a youtube video.

# TO DO
1. Recommender system (picks the next video based on evaluation result)
2. Progress tracking dashboard
3. Full system integration

___

# Guide for the TO DO
First what are we concerend here to build the recommender system are the 'youtube_transcript_app_3.py' and the 'APIs.ipynb'. The recommender system should has the following workflow:
1. LLM searches Youtube.
2. Finds a bunch of videos.
3. Looks for any video that has captions. 
4. If it finds one, it analyzes it is transcripts.
5. If not , it goes for the meta data.
6. Then the LLM picks the best fit

The recommender system's input could look like this:
```
{
  "topic": "calculus derivatives",
  "student_level": "beginner/intermediate/advanced",
  "passed": true,
  "weak_areas": ["chain rule", "implicit differentiation"],
  "preference": "videos",
  "watched_video_ids": ["abc123", "xyz789"]
}
```
Progress Tracking should store per student:
1. Video ID + title watched
2. Score per quiz (raw + percentage)
3. Passed or not
4. Evaluation text + recommendation received
5. Timestamp
6. Videos already watched (to avoid repeats)
7. Learning path so far (sequence of videos in order)

Models directories links to download:
https://drive.google.com/drive/folders/15047-E2jRbeoHLkxsvlStT7cD20-UJwo?usp=sharing
https://drive.google.com/drive/folders/1dF2Zq3i4ki-X4tmkOt4txfhd1Y9SWHjL?usp=sharing
