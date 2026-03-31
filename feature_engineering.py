import pandas as pd

#Load dataset
df = pd.read_csv("C:\\Users\\karsh\\OneDrive\\Desktop\\intern-performance-prediction\\intern_dataset (1).csv")

#Feature Engineering
df["efficiency"] = df["tasks_completed"] / df["active_days"]

df["engagement_index"] = (
    df["engagement_score"] * 0.5 +
    df["communication_score"] * 0.3 +
    df["discipline_score"] * 0.2
)

df["performance_intensity"] = df["evaluation_score"] * df["completion_rate"]

#Save new dataset
df.to_csv("C:\\Users\\karsh\\OneDrive\\Desktop\\intern-performance-prediction\\intern_dataset (1).csv", index=False)

print("Feature Engineering Done ✅")