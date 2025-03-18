import pandas as pd

def aggregate_diagnostic(y_dic: dict, agg_df: pd.DataFrame):
  tmp = []
  for key in y_dic.keys():
      if key in agg_df.index:
          c = agg_df.loc[key].diagnostic_class
          if str(c) != "nan":
            tmp.append(c)
  return list(set(tmp))