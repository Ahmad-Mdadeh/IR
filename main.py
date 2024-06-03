from flask import Flask, request,render_template
from app.Services import main

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods= ['POST'])
def submit():
    query = request.form['query']
    type_data_set= request.form['data_set']
    results,number_of_ID = main(query,type_data_set)
    return render_template('index.html',result=results,number_of_results=number_of_ID)


if __name__ == '__main__':
    app.run()

