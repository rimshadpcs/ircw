#
# Libraries
#
import pandas as pd
from flask import Flask, request, flash, render_template
from flask_paginate import Pagination, get_page_args

from Indexer import indexer

app = Flask(__name__)

dataframe = pd.read_json('/Users/rimshad/Downloads/scholar-vertical-search-engine-rimshad/scholar-vertical-search-engine-rimshad/Classification/classified.json',orient=str)
dataframe = dataframe.drop_duplicates(subset=['title', 'Authors'])
DATA = dataframe.to_dict('records')
DATA = [row for row in DATA if not (row['title'] is None)]
index = indexer.index_docs(DATA, 'title')


@app.route('/')
def home():
    if 'search' in request.args:
        searchword = request.args.get('search')
        search_results = indexer.query(index, DATA, searchword)

        for entry in search_results:
            flash(entry, 'success')
    return render_template('search.html')


@app.route('/search_results')
def search_results():
    if 'search' in request.args:
        searchword = request.args.get('search')
        search_results = indexer.query(index, DATA, searchword)
        for entry in search_results:
            flash(entry, 'success')

        # automatic pagination handling
        total = len(search_results)
        page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
        pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')

        return render_template('search_results.html',
                               search_results=search_results[offset: offset + per_page],
                               search_item=searchword,
                               page=page,
                               per_page=per_page,
                               pagination=pagination,
                               len=len)


if __name__ == '__main__':
    app.secret_key = 'secret123'
    app.run(debug=True, threaded=True)

