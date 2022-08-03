import pandas as pd
import execution.spider_execution as e
import json
import numpy as np

db = "inn_1"
sol = "answer = rooms[rooms['bedtype'] == 'king']['beds'].sum()"
actual = """[[6]]"""

con = e.connect_databse("../NLP4Code-Playground/spider/database/" + db + "/" + db +".sqlite")
df_dict = e.db_to_df_dict(con)

# Comment out after done testing
# for table_name in df_dict.keys():
#     exec(f"{table_name} = df_dict['{table_name}']")

res = e.spider_execution_py(sol, df_dict)
e.spider_answer_eq(res, json.loads(actual), True)
