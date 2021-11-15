import os
import json

def get_prefills():
	res = dict()

	session_dirs = [d for d in os.listdir() if "Session " in d]

	for dn in session_dirs:
		path = os.path.join(dn, "Preparation")

		if not os.path.isdir(path):
			continue

		k = int(dn.split("Session ")[1])
		v = [f.split(".")[0] for f in os.listdir(path) if ".pdf" == f[-4:] and len(f.split(".")[0]) == 8]

		res[k] = v

	with open("prefills.json", "w") as f:
		json.dump(res, f, sort_keys=True, indent=4)

if __name__ == "__main__":
	get_prefills()