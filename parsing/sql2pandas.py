from bs4 import BeautifulSoup
import urllib3
from sys import argv
import requests
import html

SQL2PANDAS_URL = 'https://sql2pandas.pythonanywhere.com/'

POST_HEADERS = {
    'user-agent': 'Mozilla/5.0 (X11 Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chromium/80.0.3987.160 Chrome/80.0.3987.163 Safari/537.36'
}

# Tokens to search for in HTML
HTML_CSRF_ERROR_TOKEN = 'The CSRF tokens do not match.'

HTML_PANDAS_CMD_START_TOKEN = '<pre><code class="code-default">'
HTML_PANDAS_CMD_END_TOKEN = '</code></pre>'

HTML_QUERY_FORMAT_ERROR_TOKEN_1 = 'Please check the submitted SQL syntax'
HTML_QUERY_FORMAT_ERROR_TOKEN_2 = '<div class="alert alert-warning">'

# Function error messages
CSRF_ERROR = '[sql2pandas.py] Error: unable to bypass CSRF token'
SQL_FORMAT_ERROR = '[sql2pandas.py] Error: SQL syntax incorrect or not supported'

GENERAL_ERROR = '[sql2pandas.py] Error: unknown'


# Extract converted SQL2pandas command from processed HTML string using start/end tokens
def extract_pandas_cmd(processed_html):
    if processed_html.find(HTML_CSRF_ERROR_TOKEN) >= 0:
        return CSRF_ERROR

    if processed_html.find(HTML_QUERY_FORMAT_ERROR_TOKEN_1) >= 0 and processed_html.find(HTML_QUERY_FORMAT_ERROR_TOKEN_2) >= 0:
        return SQL_FORMAT_ERROR

    temp_pieces = processed_html.split(HTML_PANDAS_CMD_START_TOKEN)
    if len(temp_pieces) < 2:
        return GENERAL_ERROR

    temp = temp_pieces[1]
    if temp.find(HTML_PANDAS_CMD_END_TOKEN) < 0:
        return GENERAL_ERROR

    temp = temp[0:temp.index(HTML_PANDAS_CMD_END_TOKEN)]
    return temp


# https://stackoverflow.com/questions/51351443/get-csrf-token-using-python-requests
# Create HTTP session to bypass CSRF token check
def make_post_request(query):
    with requests.Session() as session:
        get_response = session.get(
            SQL2PANDAS_URL, headers=POST_HEADERS, verify=False)
        soup = BeautifulSoup(get_response.text, 'lxml')

        # Pull CSRF token from HTML
        csrf_token = soup.find('input', attrs={'name': 'csrf_token'})['value']

        POST_BODY = {
            'csrf_token': csrf_token,
            'query': query,
            'submit': 'Convert'
        }

        post_response = session.post(
            SQL2PANDAS_URL, data=POST_BODY, headers=POST_HEADERS)
        raw_html = post_response.text
        # Decode HTML entities and special escaped chars
        processed_html = html.unescape(raw_html)

        # print(processed_html)
        # print(type(processed_html))
        return extract_pandas_cmd(processed_html)


# API
def sql2pandas(sql_query, should_disable_warnings=True):
    if should_disable_warnings == True:
        # Disable unsecure HTTPS request (SSL) warnings
        urllib3.disable_warnings()

    return make_post_request(sql_query)


# If run as main program script
def main():
    if len(argv) <= 1:
        print('Usage: python sandbox.py <SQL_QUERY>')
        return

    query = argv[1]
    print(sql2pandas(query))


if __name__ == '__main__':
    # Disable unsecure HTTPS request (SSL) warnings
    urllib3.disable_warnings()
    main()
