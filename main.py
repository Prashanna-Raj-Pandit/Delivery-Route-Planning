# main.py
import subprocess, sys

def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Run each experiment; adjust budgets as you like
    run([sys.executable, "experiment1.py"])
    run([sys.executable, "experiment2.py"])
    run([sys.executable, "experiment3.py"])
    print("\n✓ All experiments completed.")
