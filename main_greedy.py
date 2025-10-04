# main_greedy.py
import subprocess, sys

def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    #run([sys.executable, "experiment1_greedy.py"])
    #run([sys.executable, "experiment2_greedy.py"])
    run([sys.executable, "experiment3_greedy.py"])
    print("\n✓ All greedy experiments completed.")
