from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel
from afterlife.model import load_from_path
import torch
import os

app = FastAPI()
model_dir = os.getenv("MODEL_DIR", "models/azure-valley-12/")
device = os.getenv("DEVICE", "cpu")

device = torch.device(device)
dtype = torch.float32

model = load_from_path(model_dir)
model.to(device)
model.eval()


class Point(BaseModel):
    x: float
    y: float


class Trajectory(BaseModel):
    points: List[Point]


@app.get("/up")
def up():
    return "ok"


@torch.no_grad()
@app.post("/predict")
def predict(trajectory: Trajectory):
    points = []
    for point in trajectory.points:
        points.append([point.x, point.y])

    input = torch.tensor(points, dtype=dtype).unsqueeze(0).to(device)
    print(input.shape)
    output = model(input)

    output = output.squeeze(0).detach().cpu()
    points = []
    for items in output:
        points.append(Point(x=items[0], y=items[1]))

    return Trajectory(points=points)
