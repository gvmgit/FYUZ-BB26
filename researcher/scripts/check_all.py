"""Run every check and print one verdict.   ..\.venv\Scripts\python.exe scripts\check_all.py"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
scripts = ["check_gateway.py", "check_adapters.py", "check_agents.py", "check_deploy.py", "check_finetune.py",
           "check_research.py", "check_detector.py", "check_mobility.py",
           "check_rulesearch.py", "check_research_pipeline.py", "check_checkpoint.py", "check_evidence.py"]
failed = []
for s in scripts:
    print(f"\n===== {s} =====")
    if subprocess.run([sys.executable, str(here / s)]).returncode != 0:
        failed.append(s)
print("\n" + ("ALL CHECKS PASSED" if not failed else "FAILED: " + ", ".join(failed)))
sys.exit(1 if failed else 0)
