# Removes any characters in `chars_to_remove` from the front of `s`
def trim_front(s, chars_to_remove):
    while s[0] in chars_to_remove:
        s = s[1:]
    return s


# Removes any characters in `chars_to_remove` from the back of `s`
def trim_back(s, chars_to_remove):
    while s[-1:] in chars_to_remove:
        s = s[:-1]
    return s


# Removes characters like parentheses from front/end of `s`
def trim_front_and_back(s, char_front, char_back):
    while s[0] == char_front and s[-1:] == char_back:
        s = s[1:-1]
    return s


# Find corresponding balanced closing parenthesis for opening parenthesis at index `open_idx`
def find_closing_parenthesis(s, open_idx):
    if s[open_idx-1] != '(':
        print('[find_closing_parenthesis] input open_idx error')
        return open_idx

    idx = open_idx
    ct_open = 0
    while idx < len(s):
        if s[idx] == '(':
            ct_open += 1
        elif s[idx] == ')':
            if ct_open == 0:
                return idx
            ct_open -= 1

        idx += 1

    return -1


# Determines if first non-whitespace char in `partial_sql_query` is "SELECT"
def is_next_token_select(partial_sql_query):
    return partial_sql_query.strip().find("SELECT") == 0
