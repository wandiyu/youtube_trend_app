from functions import draw,read_figure
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

@app.route('/info')
def info():
    script,div = read_figure()
    return render_template('info.html', script=script, div=div)

    
if __name__ == '__main__':
    app.run(debug=True)
