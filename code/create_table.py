# Author: DANAIAH,NAVAYA University of JNTUK KAKINADA
# Date: 03/04/2022
# Description: This python script assumes that you already have
# a database.db file at the root of your workspace.
# This python script will CREATE a table called students 
# in the database.db using SQLite3 which will be used
# to store the data collected by the forms in this app
# Execute this python script before testing or editing this app code. 
# Open a python terminal and execute this script:
# python create_table.py

# import sqlite3

# conn = sqlite3.connect('database.db')
# print("Connected to database successfully")

# conn.execute('CREATE TABLE students (name TEXT, addr TEXT, city TEXT, zip TEXT, loginid TEXT, email EMAIL, password PASSWORD)')
# print("Created table successfully!")

# conn.close()




import sqlite3

# Connect to the database
def connect_db():
    conn = sqlite3.connect('database.db')
    print("Connected to database successfully")

    # Check if the students table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
    table_exists = cursor.fetchone()

    if not table_exists:
        # Create the students table if it doesn't exist
        conn.execute('CREATE TABLE students (name TEXT, addr TEXT, city TEXT, zip TEXT, loginid TEXT, email TEXT, password TEXT)')
        print("Created table successfully!")
    else:
        print("Table 'students' already exists.")

    # Close the connection
    conn.close()
connect_db()


