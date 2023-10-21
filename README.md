# NLSearch

### Overview
This is a project for developing an embedded-based natural language serach over images and videos.


Sketch of the project
1. For each video, extract some frames (iframes)
2. Run OpenAI’s CLIP / any other way to extract an embedding for each frame
3. Allow search ( https://simonwillison.net/2023/Sep/12/llm-clip-and-chat/ , https://www.dbreunig.com/2023/09/26/faucet-finder.html )
4. Run FAISS similarity on the frames to show similar videos
