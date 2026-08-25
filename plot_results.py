import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/IDR_results.csv")

# GNSS availability
gnss_on = df["GNSS_SIMULATED_AVAILABLE"] == 1
gnss_off = df["GNSS_SIMULATED_AVAILABLE"] == 0

plt.figure(figsize=(11, 7))

# Reference GNSS trajectory
plt.plot(
    df.loc[gnss_on, "GPS_X"],
    df.loc[gnss_on, "GPS_Y"],
    label="GNSS Reference",
    linewidth=2
)

# IDR trajectory during GNSS outage
plt.plot(
    df.loc[gnss_off, "DR_X"],
    df.loc[gnss_off, "DR_Y"],
    label="IDR During GNSS Outage",
    linewidth=3
)

# IDR trajectory after recovery
plt.plot(
    df.loc[gnss_on, "DR_X"],
    df.loc[gnss_on, "DR_Y"],
    label="IDR Fused",
    linewidth=1.5
)

# Start
plt.scatter(
    df["DR_X"].iloc[0],
    df["DR_Y"].iloc[0],
    s=80,
    label="Start"
)

# Mark outage start
if gnss_off.any():
    first_outage = df.index[gnss_off][0]

    plt.scatter(
        df.loc[first_outage, "DR_X"],
        df.loc[first_outage, "DR_Y"],
        s=100,
        marker="x",
        label="GNSS Lost"
    )

# Mark recovery
if gnss_off.any():
    last_outage = df.index[gnss_off][-1]

    plt.scatter(
        df.loc[last_outage, "DR_X"],
        df.loc[last_outage, "DR_Y"],
        s=100,
        marker="x",
        label="GNSS Recovered"
    )

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")

plt.title(
    "Intelligent Dead Reckoning — GNSS Outage Recovery"
)

plt.legend()
plt.grid(True)
plt.axis("equal")

plt.tight_layout()

plt.savefig(
    "Data/IDR_GNSS_outage.png",
    dpi=300
)

plt.show()