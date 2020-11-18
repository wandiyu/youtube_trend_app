import numpy as np,pandas as pd
import re, datetime
import sklearn
from sklearn import base
from glob import glob 
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
from statsmodels.tsa.arima.model import ARIMA
import nltk
nltk.data.path.append('./nltk_data/')

from nltk.corpus import wordnet as wn
def identify_keywords(string):
    text = word_tokenize(string.lower())
    kw = list(set([lemmatizer.lemmatize(a) for a,b in pos_tag(text) if b in ['NN','NNS']]))
    food = wn.synset('food.n.02')
    food_list = list(set([w for s in food.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
    kw = [k for k in kw if re.match(r'^\w\w+$',k) and k in food_list]
    return kw

def window_mean(x, n):
    x[np.isnan(x)] = 0
    x[x>1e6] = 0
    x1 = np.nanmean(np.array(x[:3]))
    x2 = np.nanmean(np.array(x[:4]))
    x_2 = np.nanmean(np.array(x[-4:]))
    x_1 = np.nanmean(np.array(x[-3:]))
    x_out =  [x1,x2]+[np.nanmean(np.array(x[i-n:i+n+1])) for i in range(n,len(x)-n)]+[x_2,x_1]
    return x_out

def get_kw_trend(kw):
    #kw = input()
    f = glob('data/YouTube_titles*.csv')
    f.sort()
    viewCount,likeCount,likeRatio,viewRatio,CommentSentiment = np.zeros(len(f)),np.zeros(len(f)),np.zeros(len(f)),np.zeros(len(f)),np.zeros(len(f))
    for i,ff in enumerate(f):
        f1 = pd.read_csv(ff)
        likeRatio[i] = np.nanmean(f1.loc[f1.kw==kw]['likeCount'])/np.nanmean(f1.loc[f1.kw==kw]['viewCount'])
        viewRatio[i] = np.nanmean(f1.loc[f1.kw==kw]['viewratio'])
        CommentSentiment[i] = np.nanmean(f1.loc[f1.kw==kw]['comment_sentiment'])
    viewRatio[np.isnan(viewRatio)] = 0
    like = CommentSentiment*likeRatio
    like[np.isnan(like)] = 0
    return viewRatio,like

from sklearn import base
class time_series(base.BaseEstimator,base.RegressorMixin):
    def __init__(self, linear_est,non_linear_est,residual_est):
        self.linear_est = linear_est
        self.non_linear_est = non_linear_est
        self.residual_est = residual_est
    def fit(self,x,y):
        self.x0 = x[0]
        x_days = np.c_[(x - self.x0).days]
        self.linear_est.fit(x_days,y)
        residuals = y - self.linear_est.predict(x_days)
        self.non_linear_est.fit(x,residuals)
        residuals = residuals-self.non_linear_est.predict(x)
        self.residual_est.fit(x,residuals)
        return self
    def predict(self,X):
        X_days = np.c_[(X - self.x0).days]
        return self.linear_est.predict(X_days)+\
            self.non_linear_est.predict(X)+self.residual_est.predict(X)

class residual(base.BaseEstimator,base.RegressorMixin):
    def fit(self,x,y):
        y = [i[0] for i in y.values]
        mod = ARIMA(y, order=(4,0,2))
        self.res = mod.fit()
        self.len = len(y)
        return self
    def predict(self,X):
        y_out = self.res.predict(self.len+1, self.len+len(X))
        return pd.DataFrame(y_out,index=X)
    
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
class get_trend(base.BaseEstimator,base.RegressorMixin):
    def fit(self,x,y):
        self.f = LinearRegression()
        self.f.fit(x,y)
        return self
    def predict(self,X):
        return self.f.predict(x)

def add_cycle(y,t):
    var = pd.DataFrame({'var':y}, index=t)
    var['Julian'] = var.index.to_julian_date()
    var['const'] = 1
    var['sin(mon)'] = np.sin(var['Julian'] / (365.25 /12) * 2 * np.pi)
    var['cos(mon)'] = np.cos(var['Julian'] / (365.25 /12)* 2 * np.pi)
    var['sin(3mo)'] = np.sin(var['Julian'] / (365.25 / 4) * 2 * np.pi)
    var['cos(3mo)'] = np.cos(var['Julian'] / (365.25 / 4) * 2 * np.pi)
    var['sin(yr)'] = np.sin(var['Julian'] / (365.25) * 2 * np.pi)
    var['cos(yr)'] = np.cos(var['Julian'] / (365.25) * 2 * np.pi)
    var['sin(6mon)'] = np.sin(var['Julian'] / (365.25 / 2) * 2 * np.pi)
    var['cos(6mon)'] = np.cos(var['Julian'] / (365.25 / 2) * 2 * np.pi)
    return var

class get_seasonal(base.BaseEstimator,base.RegressorMixin):
    def fit(self,t,y):
        self.var = add_cycle([x[0] for x in y.values],t)
        return self
    def predict(self,t):
        regress = sklearn.linear_model.LinearRegression().fit( 
        X=self.var[['sin(yr)', 'cos(yr)','sin(mon)', 'cos(mon)', 'sin(3mo)', 'cos(3mo)', 'sin(6mon)', 'cos(6mon)']], 
        y=self.var['var'])
        var = add_cycle(np.ones(len(t)),t)
        y_new = regress.predict(X=var[['sin(yr)', 'cos(yr)','sin(mon)', 'cos(mon)', 'sin(3mo)', 'cos(3mo)', 'sin(6mon)', 'cos(6mon)']] )
        result = pd.DataFrame(y_new,index=t)
        return result
    

from scipy import interpolate
def interp(x_new,key):
    y = [np.array([ 0.12726368,  0.26699296,  0.37837867,  0.48737254,  0.54634302,
        0.62071223,  0.80051461,  0.88242821,  0.9582498 ,  1.06622358,
        1.2621425 ,  1.38800516,  1.54655086,  1.76763256,  1.94942345,
        2.0615092 ,  2.26536604,  2.45091008,  2.71445578,  2.96758879,
        3.49499457,  3.90562174,  4.60420182,  5.97595821,  8.02422907,
       30.48433221]),
        np.array([-0.00014885,  0.        ,  0.00067785,  0.00145082,  0.00209637,
        0.00252489,  0.0029149 ,  0.00345558,  0.00391298,  0.00421178,
        0.00447534,  0.00481274,  0.00512605,  0.00536608,  0.00576777,
        0.00598555,  0.00615111,  0.00681535,  0.00723442,  0.00769467,
        0.00820823,  0.00896463,  0.00962756,  0.01082877,  0.01288933,
        0.02032245])]
    x = np.arange(0,104,4)/25
    f = interpolate.CubicSpline(y[key],x)
    result = f(x_new)
    result[result>4] = 4
    result[result<0] = 0
    return result    
    
    
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.layouts import column
from bokeh.models import Legend
from bokeh.models import FuncTickFormatter
def draw_tabs(kw,t0_new):
    x = get_kw_trend(kw)
    x2,xout = [],[]
    t0 = datetime.datetime(2019,9,8)
    t = pd.to_datetime([t0\
    +datetime.timedelta(days=7*(i)) for i in range(len(x[0]))])
    x2.append(pd.DataFrame(interp(window_mean(x[0],2),0),index=t,
                columns=['view']))
    x2.append(pd.DataFrame(interp(window_mean(x[1],2),1),index=t,
                columns=['like']))
    t_1 = pd.to_datetime([t0_new+datetime.timedelta(days=7*(i))\
                    for i in range(4)])
    for j in range(2):
        f = time_series(LinearRegression(),get_seasonal(),residual())
        f.fit(x2[j].index,x2[j])
        prediction = f.predict(t_1)
        prediction[prediction>4] = 4
        prediction[prediction<0] = 0
        xout.append(prediction)
    titles = ['views rating',
              'likes rating']
    p = []
    for i in range(2):
        p.append(figure(title=titles[i], x_axis_label= 'x', y_axis_label='y',
           x_axis_type="datetime",plot_width=700, plot_height=250,y_range=(-0.2,4.2)))
        r1=p[i].line(t,[x[0] for x in x2[i].values], line_width = 2)
        r2=p[i].line(np.append(t[-1:],t_1),np.append(x2[i].values[-1],xout[i]), 
            line_dash=(4, 4), line_color="red", 
               line_width = 2)

        r3=p[i].circle(t_1,[xx[0] for xx in xout[i].values],fill_color='red',size=5, line_color="red")
        p[i].xaxis.axis_label = 'Date'
        p[i].yaxis.axis_label = 'Popularity'
        legend = Legend(items=[
        ("History"   , [r1]),
        ("Prediction" , [r2,r3]),
            ], location="center")

        p[i].add_layout(legend, 'right')
        p[i].yaxis[0].ticker = list(range(5))
        p[i].yaxis.formatter = FuncTickFormatter(code="""
        ticks = ['','Poor','Fair','Good','Excellent']
        return ticks[tick]
        """)
    return column(p),xout

def add_prediction(x1,x2):
    x1 = min(int(np.floor(x1)),3)
    x2 = min(int(np.floor(x2)),3)
    poor = '<div class="btn-group btn-group-toggle" data-toggle="buttons"> <label class="btn btn-outline-secondary active"> <input type="radio" name="options" id="option1" checked> Poor</label> <label class="btn btn-outline-secondary"><input type="radio" name="options" id="option2"> Fair</label> <label class="btn btn-outline-secondary">  <input type="radio" name="options" id="option3"> Good </label> <label class="btn btn-outline-secondary"> <input type="radio" name="options" id="option4"> Excellent </label></div>'
    fair = '<div class="btn-group btn-group-toggle" data-toggle="buttons"> <label class="btn btn-outline-secondary"> <input type="radio" name="options" id="option1"> Poor</label> <label class="btn btn-outline-secondary active"><input type="radio" name="options" id="option2" checked> Fair</label> <label class="btn btn-outline-secondary">  <input type="radio" name="options" id="option3"> Good </label> <label class="btn btn-outline-secondary"> <input type="radio" name="options" id="option4"> Excellent </label></div>'
    good = '<div class="btn-group btn-group-toggle" data-toggle="buttons"> <label class="btn btn-outline-secondary"> <input type="radio" name="options" id="option1"> Poor</label> <label class="btn btn-outline-secondary"><input type="radio" name="options" id="option2"> Fair</label> <label class="btn btn-outline-secondary active">  <input type="radio" name="options" id="option3" checked> Good </label> <label class="btn btn-outline-secondary"> <input type="radio" name="options" id="option4"> Excellent </label></div>'
    exce = '<div class="btn-group btn-group-toggle" data-toggle="buttons"> <label class="btn btn-outline-secondary"> <input type="radio" name="options" id="option1"> Poor</label> <label class="btn btn-outline-secondary"><input type="radio" name="options" id="option2"> Fair</label> <label class="btn btn-outline-secondary">  <input type="radio" name="options" id="option3"> Good </label> <label class="btn btn-outline-secondary active"> <input type="radio" name="options" id="option4" checked> Excellent </label></div>'
    levels = [poor,fair,good,exce]
    p1 = '<h3></br>During the next month <h3>'
    p2 = '<h3> views rating: </h3>'
    p3 = '<h3> likes rating: </h3>'
    return p1+p2+levels[x1]+p3+levels[x2]
    

def add_prediction2(x1,x2):
    x1 += 0.5
    x2 += 0.5
    if x1<0:
        x1 = 0
    if x2<0:
        x2 = 0
    star_fill = '<svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-star-fill" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M3.612 15.443c-.386.198-.824-.149-.746-.592l.83-4.73L.173 6.765c-.329-.314-.158-.888.283-.95l4.898-.696L7.538.792c.197-.39.73-.39.927 0l2.184 4.327 4.898.696c.441.062.612.636.283.95l-3.523 3.356.83 4.73c.078.443-.36.79-.746.592L8 13.187l-4.389 2.256z"/></svg>'
    star_half = '<svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-star-half" fill="currentColor" xmlns="http://www.w3.org/2000/svg"> <path fill-rule="evenodd" d="M5.354 5.119L7.538.792A.516.516 0 0 1 8 .5c.183 0 .366.097.465.292l2.184 4.327 4.898.696A.537.537 0 0 1 16 6.32a.55.55 0 0 1-.17.445l-3.523 3.356.83 4.73c.078.443-.36.79-.746.592L8 13.187l-4.389 2.256a.519.519 0 0 1-.146.05c-.341.06-.668-.254-.6-.642l.83-4.73L.173 6.765a.55.55 0 0 1-.171-.403.59.59 0 0 1 .084-.302.513.513 0 0 1 .37-.245l4.898-.696zM8 12.027c.08 0 .16.018.232.056l3.686 1.894-.694-3.957a.564.564 0 0 1 .163-.505l2.906-2.77-4.052-.576a.525.525 0 0 1-.393-.288L8.002 2.223 8 2.226v9.8z"/> </svg>'
    star = '<svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-star" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M2.866 14.85c-.078.444.36.791.746.593l4.39-2.256 4.389 2.256c.386.198.824-.149.746-.592l-.83-4.73 3.523-3.356c.329-.314.158-.888-.283-.95l-4.898-.696L8.465.792a.513.513 0 0 0-.927 0L5.354 5.12l-4.898.696c-.441.062-.612.636-.283.95l3.523 3.356-.83 4.73zm4.905-2.767l-3.686 1.894.694-3.957a.565.565 0 0 0-.163-.505L1.71 6.745l4.052-.576a.525.525 0 0 0 .393-.288l1.847-3.658 1.846 3.658a.525.525 0 0 0 .393.288l4.052.575-2.906 2.77a.564.564 0 0 0-.163.506l.694 3.957-3.686-1.894a.503.503 0 0 0-.461 0z"/></svg>'
    heart_fill = '<svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-heart-fill" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M8 1.314C12.438-3.248 23.534 4.735 8 15-7.534 4.736 3.562-3.248 8 1.314z"/></svg>'
    heart_half = '<svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-heart-half" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M8 1.314C3.562-3.248-7.534 4.735 8 15V1.314z"/>  <path fill-rule="evenodd" d="M8 2.748l-.717-.737C5.6.281 2.514.878 1.4 3.053c-.523 1.023-.641 2.5.314 4.385.92 1.815 2.834 3.989 6.286 6.357 3.452-2.368 5.365-4.542 6.286-6.357.955-1.886.838-3.362.314-4.385C13.486.878 10.4.28 8.717 2.01L8 2.748zM8 15C-7.333 4.868 3.279-3.04 7.824 1.143c.06.055.119.112.176.171a3.12 3.12 0 0 1 .176-.17C12.72-3.042 23.333 4.867 8 15z"/></svg>'
    heart = '<svg width="1em" height="1em" viewBox="0 0 16 16" class="bi bi-heart" fill="currentColor" xmlns="http://www.w3.org/2000/svg"> <path fill-rule="evenodd" d="M8 2.748l-.717-.737C5.6.281 2.514.878 1.4 3.053c-.523 1.023-.641 2.5.314 4.385.92 1.815 2.834 3.989 6.286 6.357 3.452-2.368 5.365-4.542 6.286-6.357.955-1.886.838-3.362.314-4.385C13.486.878 10.4.28 8.717 2.01L8 2.748zM8 15C-7.333 4.868 3.279-3.04 7.824 1.143c.06.055.119.112.176.171a3.12 3.12 0 0 1 .176-.17C12.72-3.042 23.333 4.867 8 15z"/></svg>'
    
    if x1-int(x1)>=0.5:
        mid1 = star_half
    else:
        mid1 = star
        
    if x2-int(x2)>=0.5:
        mid2 = heart_half
    else:
        mid2 = heart
    p1 = '<h3></br>During the next month <h3>'
    p2 = '<h3> views rating: </h3>'
    p3 = '<h3> likes rating: </h3>'
        
    return p1+p2+star_fill*int(x1)+mid1+star*(5-1-int(x1))+p3+heart_fill*int(x2)+mid2+heart*(5-1-int(x2))
    
def pick_week(t0,z):
    time_series = [t0+datetime.timedelta(days=7*i) for i in range(4)]
    sumz = np.zeros(4)
    for i in range(len(z)):
        sumz += [zz[0] for zz in (z[i][1][0]+z[i][1][1]).values]
    argmax = np.argmax(sumz)
    t1 = time_series[argmax].strftime('%Y-%m-%d')
    t2 = (time_series[argmax]+datetime.timedelta(days=6)).strftime('%Y-%m-%d')
    p1 = '<h4></br>The best time to publish: <h4>'
    p2 = '<h4>'+t1+' to '+t2+'</h4>'
    return p1+p2
    

from bokeh.models.widgets import Panel
from bokeh.models.widgets import Tabs
def draw(tname):
    t0 = datetime.datetime(2020,10,25)
    kws = identify_keywords(tname)
    if not len(kws):
        return '',  {'script': '', 'div': ''},'<h3> </br></br> Sorry, we do not detect any food related keyword in your text, or corresponding video never hit top 50 over last year</h3> <a  href="/"><h3> please return </h3></a>'
    p = [draw_tabs(kw,t0) for kw in kws]
    tabs = [Panel(child=p[i][0], title=kws[i]) for i in range(len(kws))]
    layout = Tabs(tabs=tabs)
    script, div = components(layout)
    divs = div.split(' ')
    divs[1] += ' align="center"'
    div = ' '.join(divs)
    kwargs = {'script': script, 'div': div}
    kwargs['title'] = 'bokeh-with-flask'
    content = '<h3></br> We detect {} keywords: </br> {} </h3> '.format(len(kws),', '.join(kws))
    pred = add_prediction(np.mean([p[i][1][0].mean() for i in range(len(p))]), np.mean([p[i][1][1].mean() for i in range(len(p))]) )+ pick_week(t0,p)  
    return content, kwargs, pred


def select_box(kws):
    head = '<div container> <form action = "select",method="get"> <label for="nkw">Choose a keyword:</label><br>  <select id="nkw" name="nkw">'
    content = ' '.join(['<option value="{}">{}</option>'.format(kw,kw) for kw in kws])
    tail = '</select><input type="submit"> </form> </div>'
    return head+content+tail

def read_figure():
    f = open('static/div.txt','r')
    div = f.read()
    f.close()
    f = open('static/script.txt','r')
    script = f.read()
    f.close()
    return script,div
