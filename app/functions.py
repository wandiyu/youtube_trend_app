import numpy as np,pandas as pd
import re, datetime
import sklearn
from sklearn import base
from glob import glob 
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


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
    x_2 = np.nanmean(np.array(x[-4:]))
    x_1 = np.nanmean(np.array(x[-3:]))
    x_out =  [np.nanmean(np.array(x[i-n:i+n+1])) for i in range(n,len(x)-n)]+[x_2,x_1]
    return x_out

def get_kw_trend(kw):
    #kw = input()
    f = glob('data/YouTube_titles*.csv')
    f.sort()
    viewCount,likeCount,likeRatio,viewRatio,CommentSentiment = np.zeros(len(f)),np.zeros(len(f)),np.zeros(len(f)),np.zeros(len(f)),np.zeros(len(f))
    for i,ff in enumerate(f):
        f1 = pd.read_csv(ff)
        viewCount[i] = f1.loc[f1.kw==kw]['viewCount'].sum()/f1.sort_values(by='viewCount',ascending=False)['viewCount'][0]
        likeCount[i] = f1.loc[f1.kw==kw]['likeCount'].sum()
        likeRatio[i] = np.nanmean(f1.loc[f1.kw==kw]['likeratio'])
        viewRatio[i] = np.nanmean(f1.loc[f1.kw==kw]['viewratio'])
        CommentSentiment[i] = np.nanmean(f1.loc[f1.kw==kw]['comment_sentiment'])
    return viewCount,likeCount,likeRatio,viewRatio,CommentSentiment

from sklearn import base
class time_series(base.BaseEstimator,base.RegressorMixin):
    def __init__(self, linear_est,non_linear_est):
        self.linear_est = linear_est
        self.non_linear_est = non_linear_est
    def fit(self,t0,y):
        self.X = [t0+datetime.timedelta(i*7) for i in range(len(y))]
        self.linear_est.fit(self.X,y)
        residuals = y - self.linear_est.predict(self.X)
        self.non_linear_est.fit(self.X,residuals)
        return self
    def predict(self,X):
        return self.linear_est.predict(X)+self.non_linear_est.predict(X)
    
    
class get_trend(base.BaseEstimator,base.RegressorMixin):
    def fit(self,t,y):
        self.t0 = t[0]
        self.a, self.b = np.polyfit(np.arange(len(y))*7,y,1)
        return self
    def predict(self,X):
        x = np.array([(xx-self.t0).days for xx in X])
        x_trend = self.a*x+self.b
        return x_trend

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
    var['sin(day)'] = np.sin(var.index.hour / 24.0 * 2* np.pi)
    var['cos(day)'] = np.cos(var.index.hour / 24.0 * 2* np.pi)
    return var

class get_seasonal(base.BaseEstimator,base.RegressorMixin):
    def fit(self,t,y):
        self.var = add_cycle(y,t)
        return self
    def predict(self,t):
        regress = sklearn.linear_model.LinearRegression().fit( 
        X=self.var[['sin(yr)', 'cos(yr)','sin(mon)', 'cos(mon)', 'sin(3mo)', 'cos(3mo)', 'sin(day)', 'cos(day)']], 
        y=self.var['var'])
        var = add_cycle(np.ones(len(t)),t)
        return regress.predict(X=var[['sin(yr)', 'cos(yr)','sin(mon)', 'cos(mon)', 'sin(3mo)', 'cos(3mo)', 'sin(day)', 'cos(day)']] )
    

from scipy import interpolate
def interp(x_new,key):
    y = [np.array([ 0.24653481,  0.34478278,  0.46295748,  0.53465747,  0.61895277,
        0.70798916,  0.81239227,  0.9312397 ,  1.02725055,  1.15880082,
        1.33343138,  1.50881636,  1.71449206,  1.96797239,  2.23061401,
        2.55301013,  2.97730739,  3.57863134,  4.3190338 ,  5.13980625,
        6.33127236,  8.19713596, 11.98543392, 20.40736818]),
        np.array([ 0.        ,  0.01        ,  0.05        ,  0.11798617,  0.77654034,
        1.41084551,  2.18363208,  3.06330866,  4.14579782,  5.06757317,
        5.92958782,  6.83421826,  7.77092321,  8.52864653,  9.41252444,
       10.15537334, 10.89356392, 11.86310723, 12.84549323, 14.27742304,
       16.1082344 , 18.11092605, 23.95082698, 42.89498157])]
    x = np.arange(4,100,4)/20
    f = interpolate.CubicSpline(y[key],x)
    return f(x_new)
    
    
    
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.layouts import column
from bokeh.models import Legend

def draw_tabs(kw):
    kinds = [3,4]
    x = get_kw_trend(kw)
    x2,xout = [],[]
    for i in range(2):
        if i == 0:
            x2.append(interp(window_mean(x[kinds[i]],2),0))
        else:
            x2.append(interp(window_mean(x[kinds[i]]*x[2],2),1))
        t0 = datetime.datetime(2019,9,29)
        f = time_series(get_trend(),get_seasonal())
        f.fit(t0,x2[i])
        tout = [datetime.datetime(2020,10,18)+datetime.timedelta(days=7*(i)) for i in range(4)]
        xout.append(f.predict(tout))
    t = np.array(f.X,dtype=np.datetime64)
    t_1 = np.array(tout, dtype=np.datetime64)
    titles = titles = ['Do people WATCH videos related to "'+kw.upper()+'"?',
              'Do people LIKE videos related to "'+kw.upper()+'"?']
    p = []
    for i in range(2):
        p.append(figure(title=titles[i], x_axis_label= 'x', y_axis_label='y',
           x_axis_type="datetime",plot_width=700, plot_height=250,y_range=(0,5)))
        r1=p[i].line(t,x2[i], line_width = 2)
        r2=p[i].line(np.append(t[-1],t_1),np.append(x2[i][-1],xout[i]), 
            line_dash=(4, 4), line_color="red", 
               line_width = 2)
        r3=p[i].circle(t_1,xout[i],fill_color='red',size=5, line_color="red")
        p[i].xaxis.axis_label = 'Date'
        p[i].yaxis.axis_label = 'Popularity'
        legend = Legend(items=[
        ("History"   , [r1]),
        ("Prediction" , [r2,r3]),
            ], location="center")

        p[i].add_layout(legend, 'right')
        p[i].yaxis[0].ticker = list(range(6))
        p[i].ygrid[0].ticker = list(range(6))
    return column(p),xout
 

def add_prediction(x1,x2):
    x1 += 0.25
    x2 += 0.25
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
    p1 = '<h3>We expect in next month, the popularity of this video will be: <h3>'
    p2 = '<h3> in terms of view, and </h3>'
    p3 = '<h3> in terms of like </h3><br>'
        
    return p1+star_fill*int(x1)+mid1+star*(5-1-int(x1))+p2+heart_fill*int(x2)+mid2+heart*(5-1-int(x2))+p3
    
def pick_week(t0,z):
    time_series = [t0+datetime.timedelta(days=7*i) for i in range(4)]
    from functools import reduce
    if len(z) == 1:
        sumz = z[0][1][0]+z[0][1][1]
    else:
        sumz = reduce(lambda x,y: x[1][0]+x[1][1]+y[1][0]+y[1][1], z)
    argmax = np.argmax(sumz)
    t1 = time_series[argmax].strftime('%Y-%m-%d')
    t2 = (time_series[argmax]+datetime.timedelta(days=6)).strftime('%Y-%m-%d')
    p1 = '<h4>The best time next month to publish this video is over the week: <h4>'
    p2 = '<h4>'+t1+' to '+t2+'</h4>'
    return p1+p2
    

from bokeh.models.widgets import Panel
from bokeh.models.widgets import Tabs
def draw(tname):
    t0 = datetime.datetime(2020,10,11)
    kws = identify_keywords(tname)
    if not len(kws):
        return '',  {'script': '', 'div': ''},'<h3> </br></br> Sorry, we do not detect any keyword related to food in your text</h3> <a  href="/"><h3> please return </h3></a>'
    p = [draw_tabs(kw) for kw in kws]
    tabs = [Panel(child=p[i][0], title=kws[i]) for i in range(len(kws))]
    layout = Tabs(tabs=tabs)
    script, div = components(layout)
    kwargs = {'script': script, 'div': div}
    kwargs['title'] = 'bokeh-with-flask'
    content = '<h3></br></br> Detailed report</br></h3><h4> We detect {} keywords: </br> {} </h4> '.format(len(kws),', '.join(kws))
    pred = add_prediction(np.mean([p[i][1][0].mean() for i in range(len(p))]), np.mean([p[i][1][1].mean() for i in range(len(p))]) )+ pick_week(t0,p)  
    return content, kwargs, pred


def select_box(kws):
    head = '<div container> <form action = "select",method="get"> <label for="nkw">Choose a keyword:</label><br>  <select id="nkw" name="nkw">'
    content = ' '.join(['<option value="{}">{}</option>'.format(kw,kw) for kw in kws])
    tail = '</select><input type="submit"> </form> </div>'
    return head+content+tail