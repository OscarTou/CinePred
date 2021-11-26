from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from CinePred.data.featuring import *
<<<<<<< HEAD




=======
>>>>>>> c313435dad9f08bb645172542d468d86c2efd027
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"App Name": "CinePred"}
