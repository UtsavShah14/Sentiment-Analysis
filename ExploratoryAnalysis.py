import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('expand_frame_repr', False)
dataset = pd.read_csv("Tweets.csv")
print(dataset.sample(10))

val_count = dataset.Sentiment.value_counts()
plt.figure(figsize=(8, 4))
plt.bar(val_count.index, val_count.values)
plt.title("Sentiment Data Distribution")
plt.show()