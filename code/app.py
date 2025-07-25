# Author: DANAIAH, NAVAYA University JNTUK KAKAINADA
# Date: 03/4/2022
# Description: This is a Flask App that uses SQLite3 to
# execute (C)reate, (R)ead, (U)pdate, (D)elete operations

from flask import Flask
from flask import render_template
from flask import request,session, flash
import sqlite3

app = Flask(__name__)

# Home Page route
@app.route("/")
def home():
    return render_template("home.html")


def userhome():
    return render_template('user/userhome')
# Route to form used to add a new student to the database
@app.route("/enternew")
def enternew():
    return render_template("student.html")

# Route to add a new record (INSERT) student data to the database
@app.route("/addrec", methods = ['POST', 'GET'])
def addrec():
    # Data will be available from POST submitted by the form
    if request.method == 'POST':
        try:
            nm = request.form['nm']
            addr = request.form['add']
            city = request.form['city']
            zip = request.form['zip']
            loginid = request.form['loginid']
            email = request.form['email']
            password = request.form['password']

            # Connect to SQLite3 database and execute the INSERT
            with sqlite3.connect('database.db') as con:
                cur = con.cursor()
                cur.execute("INSERT INTO students (name, loginid, email,  password, addr, city, zip) VALUES (?,?,?,?,?,?,?)",(nm, loginid, email, password, addr, city, zip))

                con.commit()
                msg = "Record successfully added to database"
        except:
            con.rollback()
            msg = "Error in the INSERT"
        finally:
            con.close()
            # Send the transaction message to result.html
            return render_template('result.html',msg=msg)

# Route to SELECT all data from the database and display in a table      
@app.route('/list')
def list():
    # Connect to the SQLite3 datatabase and 
    # SELECT rowid and all Rows from the students table.
    con = sqlite3.connect("database.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT rowid, * FROM students")

    rows = cur.fetchall()
    con.close()
    # Send the results of the SELECT to the list.html page
    return render_template("list.html",rows=rows)

# Route that will SELECT a specific row in the database then load an Edit form 
@app.route("/edit", methods=['POST','GET'])
def edit():
    if request.method == 'POST':
        try:
            # Use the hidden input value of id from the form to get the rowid
            id = request.form['id']
            # Connect to the database and SELECT a specific rowid
            con = sqlite3.connect("database.db")
            con.row_factory = sqlite3.Row

            cur = con.cursor()
            cur.execute("SELECT rowid, * FROM students WHERE rowid = " + id)

            rows = cur.fetchall()
        except:
            id=None
        finally:
            con.close()
            # Send the specific record of data to edit.html
            return render_template("edit.html",rows=rows)

# Route used to execute the UPDATE statement on a specific record in the database
@app.route("/editrec", methods=['POST','GET'])
def editrec():
    # Data will be available from POST submitted by the form
    if request.method == 'POST':
        try:
            # Use the hidden input value of id from the form to get the rowid
            rowid = request.form['rowid']
            nm = request.form['nm']
            addr = request.form['add']
            city = request.form['city']
            zip = request.form['zip']
            loginid = request.form['loginid']
            email = request.form['email']
            password = request.form['password']


            # UPDATE a specific record in the database based on the rowid
            with sqlite3.connect('database.db') as con:
                cur = con.cursor()
                cur.execute("UPDATE students SET name='"+nm+"', addr='"+addr+"', city='"+city+"', zip='"+zip+"', loginid='"+loginid+"',email='"+email+"', password='"+password+"' WHERE rowid="+rowid)

                con.commit()
                msg = "Record successfully edited in the database"
        except:
            con.rollback()
            msg = "Error in the Edit: UPDATE students SET name="+nm+", addr="+addr+", city="+city+", zip="+zip+" loginid="+loginid+",email="+email+", password="+password+" WHERE rowid="+rowid

        finally:
            con.close()
            # Send the transaction message to result.html
            return render_template('result.html',msg=msg)

# Route used to DELETE a specific record in the database    
@app.route("/delete", methods=['POST','GET'])
def delete():
    if request.method == 'POST':
        try:
             # Use the hidden input value of id from the form to get the rowid
            rowid = request.form['id']
            # Connect to the database and DELETE a specific record based on rowid
            with sqlite3.connect('database.db') as con:
                    cur = con.cursor()
                    cur.execute("DELETE FROM students WHERE rowid="+rowid)

                    con.commit()
                    msg = "Record successfully deleted from the database"
        except:
            con.rollback()
            msg = "Error in the DELETE"

        finally:
            con.close()
            # Send the transaction message to result.html
            return render_template('result.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()
        loginid = request.form['loginid']
        password = request.form['password']
        print(loginid, password)
        query = "SELECT loginid, password FROM students WHERE loginid=? AND password=?"
        cursor.execute(query, (loginid, password))
        results = cursor.fetchall()        
        if not results:
            error_message = 'Invalid login credentials. Please try again.'
            return render_template('login.html', error=error_message)
        else:
            return render_template('user/userhome.html')
    return render_template('login.html')
import pandas as pd
import numpy as np
import os
from flask import current_app
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
# path = os.path.join(current_app.root_path, 'media', 'ElectricCarData_Clean_Me.csv')    
# path1 = pd.read_csv(path)

@app.route('/dataset')
def dataset():   
    path = os.path.join(current_app.root_path, 'media', 'ElectricCarData_Clean_Me.csv')    
    path1 = pd.read_csv(path)
    data = path1.to_html()    
    return render_template('user/dataset.html',data=data)


@app.route('/training')
def training():
    path = os.path.join(current_app.root_path, 'media', 'ElectricCarData_Modified.csv')    #new 7th aug
    
    df = pd.read_csv(path)
    # df['FullName'] = df['Brand'] + '-' + df['Model']
    df_1 = df.loc[df['PriceEuro'] <= 50000]
    df_2 = df.loc[df['PriceEuro'] > 50000]
    t1 = 'Less than 50,000 Euros' 
    t2 = 'More than 50,000 Euros'
    # def power_train(dataframe): 
    #     sns.countplot(x= dataframe['PowerTrain'])
    #     plt.title('Count Plot of Powertrain', fontsize = 20)
    #     plt.xlabel('Power Train', fontsize = 15)
    #     plt.ylabel('Count', fontsize = 15)
    # power_train(df)
    # def bodystyle(dataframe):
    #     plt.figure(figsize=(10, 5))
    #     sns.countplot(x= 'BodyStyle', data= dataframe, hue='PowerTrain')
    #     plt.title('Count plot of Body Style', fontsize= 20)
    #     plt.xlabel('Body Style', fontsize= 15)
    #     plt.ylabel('Count', fontsize= 15)
    #     plt.show()  
    # bodystyle(df)
    # def range(dataframe, price):
    #     plt.figure(figsize=(20,5))
    #     sns.set_theme(style="whitegrid")
    #     sns.barplot(x='FullName', y='Range_Km', data=dataframe, hue='PowerTrain')
    #     plt.title('''Range(Km) of EV's costing {} '''.format(price), fontsize=20)
    #     plt.ylabel('Range (Km)', fontsize=15)
    #     plt.xlabel('Model', fontsize=15)
    #     plt.xticks(rotation=90)
    #     plt.show()
    # # Bar Graphs
    # range(df_1, t1)
    # range(df_2, t2)
    # def range_batterypack(dataframe, text):
    #     fig = plt.figure(figsize=(20,5))
    #     ax1 = plt.subplot()
    #     ax1.bar(dataframe['FullName'], dataframe['Range_Km'],label= 'Range (Km)', color= 'steelblue')
    #     plt.legend(loc= 'upper left', bbox_to_anchor = (0, 1.105))
    #     ax2 = ax1.twinx()
    #     ax2.scatter(dataframe['FullName'], dataframe['Battery_Pack Kwh'], label= 'Battery Pack', color = 'black')
            
    #     plt.title('''RANGE (Km) vs BATTERY PACK CAPACITY (KwH) of EV's costing {}'''.format(text), fontsize= 20)
    #     ax1.set_xlabel('Models', size = 20)
    #     ax1.set_ylabel('Range (Km)', color = 'steelblue', size = 20)
    #     ax2.set_ylabel('Battery Pack Capacity (Kwh)', color= 'black', size= 20)
    #     plt.legend(loc= 'upper left', bbox_to_anchor = (0, 1))
    #     ax1.set_xticklabels(df_1['FullName'], rotation = 'vertical')
    #     plt.show()
    # range_batterypack(df_1, t1)
    # range_batterypack(df_2, t2)
    # def acc(dataframe, text):
    #     plt.figure(figsize=(20,5))
    #     sns.set_theme(style="whitegrid")
    #     sns.barplot(x='FullName', y='AccelSec', data=dataframe, hue='PowerTrain')
    #     plt.title('''Acceleration 0-100 Km of EV's costing {}'''.format(text), fontsize=20)
    #     plt.ylabel('Acceleration (seconds)')
    #     plt.xlabel('Model')
    #     plt.xticks(rotation=90)
    #     plt.show()
    # # Acceleration
    # acc(df_1, t1)
    # acc(df_2, t2)
        
    # def range_price(dataframe, text):
    #     fig = plt.figure(figsize=(20,5))
    #     ax1 = plt.subplot()
    #     ax1.bar(dataframe['FullName'], dataframe['Range_Km'],label= 'Range (Km)', color= 'steelblue')
    #     plt.legend(loc= 'upper left', bbox_to_anchor = (0, 1.1))
    #     ax2 = ax1.twinx()
    #     ax2.scatter(dataframe['FullName'], dataframe['PriceEuro'], label= 'Price', color = 'black')
    #     plt.title('''RANGE (Km) vs PRICE (Euros)of EV's costing {}'''.format(text), fontsize= 20)
    #     ax1.set_xlabel('Models', size = 20)
    #     ax1.set_ylabel('Range (Km)', color = 'steelblue', size = 20)
    #     ax2.set_ylabel('Price (Euros)', color= 'black', size= 20)
    #     plt.legend(loc= 'upper left', bbox_to_anchor = (0, 1))
    #     ax1.set_xticklabels(df_1['FullName'], rotation = 'vertical')
    #     plt.show()
    # # price vs acceleration
    # range_price(df_1, t1)
    # range_price(df_2, t2)    
    
    # def range_efficiency(dataframe, text):
    #     fig = plt.figure(figsize=(20,5))
    #     ax1 = plt.subplot()
    #     ax1.bar(dataframe['FullName'], dataframe['Range_Km'],label= 'Range (Km)', color= 'darkseagreen')
    #     plt.legend(loc= 'upper left', bbox_to_anchor = (0, 1.1))
    #     ax2 = ax1.twinx()
    #     ax2.scatter(dataframe['FullName'], dataframe['Efficiency_WhKm'], label= 'Price', color = 'black')
    #     plt.title('''RANGE (Km) vs Efficiency (Wh/km)of EV's costing {}'''.format(text), fontsize= 20)
    #     ax1.set_xlabel('Models', size = 20)
    #     ax1.set_ylabel('Range (Km)', color = 'darkseagreen', size = 20)
    #     ax2.set_ylabel('Efficiency (Wh/Km)', color= 'black', size= 20)
    #     plt.legend(loc= 'upper left', bbox_to_anchor = (0, 1))
    #     ax1.set_xticklabels(df_1['FullName'], rotation = 'vertical')
    #     plt.show()
    # # range vs efficiency
    # range_efficiency(df_1, t1)
    # range_efficiency(df_2, t2)    
    # def fastcharge(dataframe, price):
    #     plt.figure(figsize=(20, 5))
    #     sns.set_theme(style="whitegrid")
    #     sns.barplot(x='FullName', y='FastCharge_KmH', data=dataframe, color='lightslategrey')
    #     plt.title('''Fast Charging of EV's costing {} '''.format(price), fontsize=20)
    #     plt.ylabel('Charging Capacity (kmH)', fontsize=15)
    #     plt.xlabel('Model', fontsize=15)
    #     plt.xticks(rotation=90)
    #     plt.show()

    # # Fast charge Data
    # fastcharge(df_1, t1)
    # fastcharge(df_2, t2) 
    # df
    # df = df.drop(['Brand','Model','FullName'],axis=1)
    # df
    X = df[[x for x in df.columns if x != 'PriceEuro']]  # new 7th aug
    Y = df['PriceEuro']  # Target (what we want to predict)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # # Number of features to consider at every split
    # max_features = ['auto', 'sqrt']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    # # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #             'max_features': max_features,
    #             'max_depth': max_depth,
    #             'min_samples_split': min_samples_split,
    #             'min_samples_leaf': min_samples_leaf,
    #             'bootstrap': bootstrap}
    # print(random_grid)
    # {'bootstrap': [True, False],
    # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    # 'max_features': ['auto', 'sqrt'],
    # 'min_samples_leaf': [1, 2, 4],
    # 'min_samples_split': [2, 5, 10],
    # 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    # rf = RandomForestRegressor()
    # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
       
    # def evaluate(model, test_features, test_labels):
    #     predictions = model.predict(test_features)
    #     errors = abs(predictions - test_labels)
    #     mape = 100 * np.mean(errors / test_labels)
    #     accuracy = 100 - mape
    #     print('Model Performance')
    #     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    #     print('Accuracy = {:0.2f}%.'.format(accuracy)) 
    # base_model = RandomForestRegressor(n_estimators = 1600, random_state = 42)
    # base_model.fit(X, Y)
    # base_accuracy = evaluate(base_model, X, Y)
    # # Perform evaluation directly without using the evaluate function
    # predictions = base_model.predict(X)
    # errors = abs(predictions - Y)
    # mape = 100 * np.mean(errors / Y)
    # accuracy = 100 - mape
    
    # # Save the trained model
    # import joblib
    # joblib.dump(base_model, 'trained_model.joblib')
    
    # # Print model performance metrics
    # print('Model Performance')
    # print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    # print('Accuracy = {:0.2f}%.'.format(accuracy))
    # return render_template('user/training.html',accuracy=accuracy,errors=mape)
    # FIX 3: Handle missing values properly
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # FIX 4: Train on training data only
    base_model = RandomForestRegressor(n_estimators=1600, random_state=42)
    base_model.fit(X_train_imputed, y_train)  # Train on training set
    
    # FIX 5: Evaluate on test data (unseen data)
    predictions = base_model.predict(X_test_imputed)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    
    # Save both model and imputer
    import joblib
    joblib.dump(base_model, 'trained_model.joblib')
    joblib.dump(imputer, 'data_imputer.joblib')  # Important: save the imputer too!
    
    print('Model Performance')
    print('Average Error: {:0.4f} euros.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return render_template('user/training.html', accuracy=accuracy, errors=mape)
    
# @app.route('/predication',methods=['GET', 'POST'])
# def predication():
#      if request.method == 'POST':
#         AccelSec = request.form.get("AccelSec")
#         TopSpeed_KmH = request.form.get('TopSpeed_KmH')
#         Range_Km = request.form.get('Range_km')
#         Battery_Pack_Kwh = request.form.get('Battery_Pack_Kwh') 
#         Efficiency_WhKm = request.form.get('Efficiency_WhKm')
#         FastCharge_KmH = request.form.get('FastCharge_KmH')
#         RapidCharge = request.form.get('RapidCharge')
#         PowerTrain = request.form.get('PowerTrain')
#         PlugType = request.form.get('PlugType')
#         BodyStyle = request.form.get('BodyStyle')  
#         Segment = request.form.get('Segment')
#         Seats = request.form.get('Seats')         
#         path = os.path.join(current_app.root_path, 'media','ElectricCarData_Modified.csv')    
#         df = pd.read_csv(path)   
        
#         column_mappings = {
#             'RapidCharge':{'Yes':1,'No':0},
#             'PlugType':{'Type 2 CCS':1,'Type 2':2,'Type 2 CHAdeMO':3},
#             'PowerTrain':{'AWD':1,'FWD':2,'RWD':3},
#             'BodyStyle':{'SUV':1,'Hatchback':2,'Sedan':3,'Liftback':4,'Pickup':5,'Cabrio':6,'SPV':7},
#             'Segment':{'C':1,'B':2,'D':3,'F':4,'E':5,'A':6,'N':7}
#         }              
#         df.replace(column_mappings, inplace=True)
#         print(df)        
#         # Compute the correlation matrix       
#         X = df[[x for x in df.columns if x!='PriceEuro']]
#         print(X)
#         # X = pd.get_dummies(X)
#         Y = df['PriceEuro']
        
#         # Importing necessary libraries
#         from sklearn.impute import SimpleImputer
#         from sklearn.ensemble import RandomForestRegressor
#         from sklearn.model_selection import train_test_split

#         # Impute missing values
#         imputer = SimpleImputer(strategy='mean')
#         X_imputed = imputer.fit_transform(X)
        
        
#         # Splitting the data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(X_imputed, Y, test_size=0.2, random_state=42)

#         # Initializing the Random Forest regressor
#         rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

#         # Training the model
#         rf_regressor.fit(X_train, y_train)
#         X_test = [[AccelSec,TopSpeed_KmH,Range_Km,Battery_Pack_Kwh,Efficiency_WhKm, FastCharge_KmH, RapidCharge, PowerTrain, PlugType, BodyStyle, Segment, Seats]]        
        

#         # Making predictions on the test set
#         predictions = rf_regressor.predict(X_test)        
#         #new features               
#         return render_template('user/predication.html',y_pred=predictions)     
#      return render_template('user/predication.html')     
# Importing necessary libraries
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from flask import request
@app.route('/predication', methods=['GET', 'POST'])
def predication():
    if request.method == 'POST':
        # Get form inputs (same as before)
        def to_float_or_none(value):
            return float(value) if value is not None and value.strip() else None        
        
        AccelSec = to_float_or_none(request.form.get("AccelSec"))
        TopSpeed_KmH = to_float_or_none(request.form.get('TopSpeed_KmH'))
        Range_Km = to_float_or_none(request.form.get('Range_km'))
        Battery_Pack_Kwh = to_float_or_none(request.form.get('Battery_Pack_Kwh'))
        Efficiency_WhKm = to_float_or_none(request.form.get('Efficiency_WhKm'))
        FastCharge_KmH = to_float_or_none(request.form.get('FastCharge_KmH'))
        RapidCharge = to_float_or_none(request.form.get('RapidCharge'))
        PowerTrain = to_float_or_none(request.form.get('PowerTrain'))
        PlugType = to_float_or_none(request.form.get('PlugType'))
        BodyStyle = to_float_or_none(request.form.get('BodyStyle'))
        Segment = to_float_or_none(request.form.get('Segment'))
        Seats = to_float_or_none(request.form.get('Seats'))
        
        # Load the trained model and imputer
        import joblib
        rf_regressor = joblib.load('trained_model.joblib')
        imputer = joblib.load('data_imputer.joblib')
        
        # Prepare input for prediction
        X_test = np.array([[AccelSec, TopSpeed_KmH, Range_Km, Battery_Pack_Kwh, Efficiency_WhKm, FastCharge_KmH,
                            RapidCharge, PowerTrain, PlugType, BodyStyle, Segment, Seats]])
        
        # Apply same preprocessing as training
        X_test_imputed = imputer.transform(X_test)
        
        # Make prediction
        predictions = rf_regressor.predict(X_test_imputed)
        
        return render_template('user/predication.html', y_pred=predictions[0])
    
    return render_template('user/predication.html')

if __name__ == '__main__':
    app.run(debug=True,port=8000)