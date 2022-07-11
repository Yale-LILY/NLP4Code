import pandas as pd
import execution.spider_execution as e
import json
import numpy as np

con = e.connect_databse("../NLP4Code-Playground/spider/database/college_2/college_2.sqlite")
df_dict = e.db_to_df_dict(con)

# Comment out after done testing
# for table_name in df_dict.keys():
#     exec(f"{table_name} = df_dict['{table_name}']")

res = e.spider_execution_py("t1 = pd.merge(instructor, department, left_on='dept_name', right_on='dept_name')\nt2 = t1.sort_values(by='budget', ascending=False)\nanswer = [t2['salary'].mean(), t2.shape[0]]", df_dict)
e.spider_answer_eq(res, json.loads('[[77600.18819999999, 50]]'), False)
