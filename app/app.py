from functions import draw
from flask import Flask, render_template, request, redirect
import pandas as pd, numpy as np
import urllib
import simplejson as json

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/graph')
def graph():
    tname = request.args.get('kw')
    content, kwargs, pred = draw(tname)
    return render_template('showresult.html', content=content,pred = pred, **kwargs)  
    
if __name__ == '__main__':
    app.run(debug=True)
