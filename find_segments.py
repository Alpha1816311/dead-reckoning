import pandas as pd
import math

d = pd.read_csv("Data/S-S1.csv", engine="python", encoding="cp1252")

lat = pd.to_numeric(d.iloc[:,0], errors="coerce").values
lon = pd.to_numeric(d.iloc[:,1], errors="coerce").values
t = pd.to_numeric(d.iloc[:,7], errors="coerce").values / 1000.0

R = 6371000.0

def distance(a, b):
    total = 0
    for i in range(a + 1, b):
        if any(pd.isna([lat[i-1], lon[i-1], lat[i], lon[i]])):
            continue
        p1 = math.radians(lat[i-1])
        p2 = math.radians(lat[i])
        dp = math.radians(lat[i] - lat[i-1])
        dl = math.radians(lon[i] - lon[i-1])
        h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        total += 2 * R * math.asin(math.sqrt(h))
    return total

results = []

for start in range(0, int(t[-1])-30, 30):
    a = (abs(t-start)).argmin()
    b = (abs(t-(start+30))).argmin()
    results.append((distance(a,b), start))

results.sort(reverse=True)

print("Dataset duration:", round(t[-1],2), "seconds")
print("\nTop moving 30-second segments:")
for dist, start in results[:10]:
    print(f"Start {start:5.0f}s -> {start+30:5.0f}s : {dist:8.2f} m")
