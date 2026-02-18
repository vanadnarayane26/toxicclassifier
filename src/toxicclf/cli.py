import typer

from toxicclf.train import run_training
from toxicclf.inference.predict import run_inference

app = typer.Typer()

@app.command()
def train(data: str = typer.Option(None, "--data", help="Path to the training data CSV file")):
    run_training(data)
    
@app.command()
def predict(text: str = typer.Option(None, "--text", help="Text to predict on")):
    run_inference(text)
    

if __name__ == "__main__":
    app()
