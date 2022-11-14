from tkinter import *

from preprocess import CATEGORY_INVERSED
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from tk import *
import pickle

root = Tk()
root.geometry("400x400")
root.title("BAYES Classification")
root.config(bg="blue")
input_variable = StringVar()
entry1 = Entry(root, fg="blue", font="courrier", textvariable=input_variable)


def btn2():
    root.destroy()


def btn1():
    print("Loading counting vector...")
    count_vector = pickle.load(open("results/count_vector.pickle", "rb"))

    print("Loading model...")
    naive_bayes = pickle.load(open("results/naive_bayes.pickle", "rb"))

    reviews = [input_variable.get()]
    x = count_vector.transform(reviews)
    result = naive_bayes.predict(x)
    result = [CATEGORY_INVERSED[res] for res in result]
    lbl = Label(root, text=result, fg="red", font="courrier", bg="blue")
    lbl.place(x=100, y=150)
    # print(result)


button1 = Button(root, text="PREDICT", command=btn1, fg="blue", font="courrier")
button2 = Button(root, text="QUIT", command=btn2, fg="blue", font="courrier")
lbl1 = Label(root, text="Enter a text for prediction", fg="white", bg="blue", font="courrier")
lbl1.place(x=100, y=50)
entry1.place(x=100, y=100)
button1.place(x=50, y=200)
button2.place(x=250, y=200)
root.mainloop()
