from flask import Flask, redirect, url_for, request, jsonify, render_template
from getData import *
from json import *

app = Flask(__name__)


@app.route('/')
def success():
    # lists=getdata("computer")
    return render_template('form.html')


''''@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))
@app.route('/postmethod', methods = ['POST'])
def post_javascript_data():
    jsdata = request.form['canvas_data']
    #unique_id = create_csv(jsdata)
    #params = { 'uuid' : unique_id }
    #return jsonify(params)
    #lists=getdata(jsdata)
    return jsdata'''


@app.route('/signUpUser', methods=['POST', 'GET'])
def signUpUser():
    user = request.form['query']
    print(user)
    if user:
        lists = getdata(user)
        if(lists=="keyword not present in any file"):
            l=[]
            l.append(lists)
            l.append("0")
            l.append("0")
            l.append("0")
            p=[]
            p.append(l)
            return jsonify({'name': p})
        #response = jsonify({'name': lists})
        return jsonify({'name': lists})
    # password = request.form['password'];
    return jsonify({'error': 'mis'})

'''@app.route('/signUpUser2', methods=['POST', 'GET'])
def signUpUser2():
    user = request.form['query']
    print(user)
    if user:
        lists = getdata(user)
        if(lists=="keyword not present in any file"):
            l=[]
            l.append(lists)
            l.append("0")
            l.append("0")
            l.append("0")
            p=[]
            p.append(l)
            return jsonify({'name': p})
        #response = jsonify({'name': lists})
        return jsonify({'name': lists})
    # password = request.form['password'];
    return jsonify({'error': 'mis'})'''
@app.route('/signUpUser2', methods=['POST', 'GET'])
def signUpUser2():
    user = request.form['query']
    feedback_files=[int(i,10) for i in user.split(" ")]
    search=request.form['search']
    print(search)
    print(feedback_files)
    if user:
        lists = getexploreddata(search,feedback_files)
        if(lists=="keyword not present in any file"):
            l=[]
            l.append(lists)
            l.append("0")
            l.append("0")
            l.append("0")
            p=[]
            p.append(l)
            return jsonify({'name': p})
        #response = jsonify({'name': lists})
        return jsonify({'name': lists})
    # password = request.form['password'];
    return jsonify({'error': 'mis'})


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    app.run(debug=True)
