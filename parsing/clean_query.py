import re
from helpers import trim_front_and_back


# sql2pandas requires single quotes in SQL queries
def replace_quotes(sql_query):
    return sql_query.replace("\"", "\'")


# Remove extra spaces
def remove_consecutive_spaces(sql_query):
    sql_query = sql_query.strip()
    sql_query = re.sub(r"\s+", " ", sql_query)
    sql_query = re.sub(r"\( ", "(", sql_query)
    return sql_query


# Add semi-colon at end of SQL query for consistency
def add_semicolon(sql_query):
    return sql_query if sql_query[-1:] == ";" else sql_query + ";"


# Basic string preprocessing/cleanup for SQL queries
def basic_clean_query(sql_query):
    sql_query = replace_quotes(sql_query)
    sql_query = remove_consecutive_spaces(sql_query)
    # TODO: ensure balance for front/back parentheses
    sql_query = trim_front_and_back(sql_query, "(", ")")

    sql_query = add_semicolon(sql_query)
    return sql_query
