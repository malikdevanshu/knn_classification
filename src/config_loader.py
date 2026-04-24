from pathlib import Path
import yaml

"""default = Path.home() / "Downloads"
for p in default.glob("*FINAL"):
    target = p

#for i in target.iterdir():
    #print(i)

acces = target / "DATA"

for j in acces.iterdir():
    print(j)

final = acces / "sonar.all-data.csv"

with open(final) as f:
    print(f.read())

src = final
dst = Path("data").resolve()
shutil.copy(src, dst)"""

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
