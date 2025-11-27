import pandas as pd


label_path = "/home/cwinkelmann/work/Herdnet/data/floreana_all/val/herdnet_format.csv"
df_all = pd.read_csv(label_path)

# remove everything which is not iguana_point
df_point = df_all[df_all["species"] == "iguana_point"]

df_point.to_csv(label_path, index=False)