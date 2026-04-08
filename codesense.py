import click, torch, os, subprocess
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

console = Console()
MODEL_PATH = "./model"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

def score_code(code, tokenizer, model):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    bug_score = int(probs[1].item() * 100)
    quality = 100 - bug_score
    complexity = min(100, max(10, len(code.split('\n')) * 3))
    return bug_score, quality, complexity

def make_bar(score, width=20):
    filled = int(score / 100 * width)
    color = "green" if score > 70 else "yellow" if score > 40 else "red"
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] {score}/100"

@click.command()
@click.argument("files", nargs=-1)
def main(files):
    console.print(Panel("[bold green]CodeSense[/bold green] · DL-powered code reviewer", expand=False))

    if not files:
        result = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True)
        files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py") and os.path.exists(f)]

    if not files:
        console.print("[yellow]No Python files staged. Run: git add <file> first.[/yellow]")
        return

    tokenizer, model = load_model()

    for filepath in files:
        with open(filepath) as f:
            code = f.read()

        bug_score, quality, complexity = score_code(code, tokenizer, model)

        console.print(f"\n[bold blue]{filepath}[/bold blue]")
        console.print(f"  Quality    {make_bar(quality)}")
        console.print(f"  Bug risk   {make_bar(100 - bug_score)}")
        console.print(f"  Complexity {make_bar(100 - min(complexity, 99))}")

        if bug_score > 60:
            console.print(f"  [red]⚠  High bug risk detected — review before pushing[/red]")
        else:
            console.print(f"  [green]✓  Looks clean[/green]")

    console.print("\n[dim]CodeSense scan complete.[/dim]\n")

if __name__ == "__main__":
    main()
