"""Build the zip to upload for an AWS CodeBuild deploy (deploy/codebuild/buildspec.yml at the zip root).

    ..\\.venv\\Scripts\\python.exe deploy\\make_package.py      ->  deploy\\atr_aws_package.zip

Contains only what the container image and the CDK app need: the researcher code, the trained twin, the simulation data
the twin reads, and MAVERIC's radp package. No virtualenv, runs, videos or local model downloads.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]              # the Hackathon folder (Docker build context)
OUT = Path(__file__).resolve().parent / "atr_aws_package.zip"
INCLUDE = ["researcher/atr", "researcher/deploy", "researcher/console", "researcher/policies",
           "researcher/models/twin_model_map.pickle",
           "data/sim_data/sim_data/3cell", "maveric-1.0.0 1/maveric-1.0.0/radp"]
SKIP_PARTS = {"__pycache__", "cdk.out", "cdk.out.p1", "cdk.out.p2", "node_modules", ".git"}


def files():
    for inc in INCLUDE:
        p = ROOT / inc
        if not p.exists():
            raise SystemExit(f"missing {p} (run `atr.cli build-twin` if it is the twin model)")
        for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
            parts = f.relative_to(ROOT).parts
            if (f.is_file() and not (set(parts) & SKIP_PARTS) and not any(p.startswith("cdk.out") for p in parts)
                    and f != OUT and f.suffix != ".zip" and f.name not in ("outputs.json", "atr_aws.env", "aws_env.ps1")):
                yield f


def main():
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "researcher/deploy/codebuild/buildspec.yml", "buildspec.yml")   # CodeBuild looks at the root
        for f in files():
            z.write(f, f.relative_to(ROOT).as_posix())
            n += 1
    print(f"{OUT}  ({n} files, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
