
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse
import datetime
import re
import string

from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,impact_ratio_model,rail_delay_model,rail_delay_prediction_model,detection_accuracy

def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "SProvider" and password =="SProvider":
            rail_delay_prediction_model.objects.all().delete()
            rail_delay_prediction_model.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')


def viewtreandingquestions(request,chart_type):
    dd = {}
    pos,neu,neg =0,0,0
    poss=None
    topic = rail_delay_prediction_model.objects.values('ratings').annotate(dcount=Count('ratings')).order_by('-dcount')
    for t in topic:
        topics=t['ratings']
        pos_count=rail_delay_prediction_model.objects.filter(topics=topics).values('names').annotate(topiccount=Count('ratings'))
        poss=pos_count
        for pp in pos_count:
            senti= pp['names']
            if senti == 'positive':
                pos= pp['topiccount']
            elif senti == 'negative':
                neg = pp['topiccount']
            elif senti == 'nutral':
                neu = pp['topiccount']
        dd[topics]=[pos,neg,neu]
    return render(request,'SProvider/viewtreandingquestions.html',{'object':topic,'dd':dd,'chart_type':chart_type})

def Find_Impact_Ratio_Delay(request): # Search
    impact_ratio_model.objects.all().delete()
    ratio = ""
    kword = 'More Late'
    print(kword)
    obj = rail_delay_prediction_model.objects.all().filter(Q(impact=kword))
    obj1 = rail_delay_prediction_model.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        impact_ratio_model.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Average Late'
    print(kword1)
    obj1 = rail_delay_prediction_model.objects.all().filter(Q(impact=kword1))
    obj11 = rail_delay_prediction_model.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        impact_ratio_model.objects.create(names=kword1, ratio=ratio1)

    ratio12 = ""
    kword12 = 'Less Late'
    print(kword12)
    obj12 = rail_delay_prediction_model.objects.all().filter(Q(impact=kword12))
    obj112 = rail_delay_prediction_model.objects.all()
    count12 = obj12.count();
    count112 = obj112.count();
    ratio12 = (count12 / count112) * 100
    if ratio12 != 0:
        impact_ratio_model.objects.create(names=kword12, ratio=ratio12)
    obj = impact_ratio_model.objects.all()
    return render(request, 'SProvider/Find_Impact_Ratio_Delay.html', {'objs': obj, 'count': ratio1})

def View_Impact_Prediction_Details(request):

    obj = rail_delay_prediction_model.objects.all()
    return render(request, 'SProvider/View_Impact_Prediction_Details.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = rail_delay_prediction_model.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def negativechart(request,chart_type):
    dd = {}
    pos, neu, neg = 0, 0, 0
    poss = None
    topic = rail_delay_prediction_model.objects.values('ratings').annotate(dcount=Count('ratings')).order_by('-dcount')
    for t in topic:
        topics = t['ratings']
        pos_count = rail_delay_prediction_model.objects.filter(topics=topics).values('names').annotate(topiccount=Count('ratings'))
        poss = pos_count
        for pp in pos_count:
            senti = pp['names']
            if senti == 'positive':
                pos = pp['topiccount']
            elif senti == 'negative':
                neg = pp['topiccount']
            elif senti == 'nutral':
                neu = pp['topiccount']
        dd[topics] = [pos, neg, neu]
    return render(request,'SProvider/negativechart.html',{'object':topic,'dd':dd,'chart_type':chart_type})


def charts(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = impact_ratio_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def Train_View_Rail_DataSets_Details(request):
    detection_accuracy.objects.all().delete()
    df = pd.read_csv('Rail_DataSets.csv')
    df
    df.columns
    df.rename(columns={'distruption_reason': 'dreason', 'distruption_time': 'dtime'}, inplace=True)

    def apply_results(results):
        if (results>=60):
            return 0  # More Late
        elif results >= 30 and results <= 60:
            return 1  # Average Late
        elif results >= 10 and results <= 30:
            return 2  # Less Late

    df['results'] = df['dtime'].apply(apply_results)

    X = df['dreason']
    y = df['results']

    cv = CountVectorizer()

    x = cv.fit_transform(X)

    models = []
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    X_train.shape, X_test.shape, y_train.shape

    print("Naive Bayes")

    from sklearn.naive_bayes import MultinomialNB

    NB = MultinomialNB()
    NB.fit(X_train, y_train)
    predict_nb = NB.predict(X_test)
    naivebayes = accuracy_score(y_test, predict_nb) * 100
    print("ACCURACY")
    print(naivebayes)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_nb))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_nb))
    detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)

    # SVM Model
    print("SVM")
    from sklearn import svm

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print("ACCURACY")
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

    print("Logistic Regression")

    from sklearn.linear_model import LogisticRegression

    reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, y_pred) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))
    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

    print("Decision Tree Classifier")
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtcpredict = dtc.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, dtcpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, dtcpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, dtcpredict))
    detection_accuracy.objects.create(names="Decision Tree Classifier", ratio=accuracy_score(y_test, dtcpredict) * 100)

    print("SGD Classifier")
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(loss='hinge', penalty='l2', random_state=0)
    sgd_clf.fit(X_train, y_train)
    sgdpredict = sgd_clf.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, sgdpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, sgdpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, sgdpredict))
    detection_accuracy.objects.create(names="SGD Classifier", ratio=accuracy_score(y_test, sgdpredict) * 100)

    print("KNeighborsClassifier")
    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier()
    kn.fit(X_train, y_train)
    knpredict = kn.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, knpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, knpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, knpredict))
    detection_accuracy.objects.create(names="KNeighborsClassifier", ratio=accuracy_score(y_test, knpredict) * 100)

    labeled = 'labeled_data.csv'
    df.to_csv(labeled, index=False)
    df.to_markdown

    se=''
    obj1 = rail_delay_model.objects.values('names','rail_name','rail_type','departure_place','destination','departure_date','departure_time',
    'arrival_date',
    'arrival_time',
    'distruption_place_name',
    'distruption_reason',
    'distruption_time',
    'actual_arrival_time'
    )

    rail_delay_prediction_model.objects.all().delete()
    for t in obj1:
        names= t['names']
        rail_name=t['rail_name']
        rail_type=t['rail_type']
        departure_place=t['departure_place']
        destination=t['destination']
        departure_date=t['departure_date']
        departure_time=t['departure_time']
        arrival_date=t['arrival_date']
        arrival_time=t['arrival_time']
        distruption_place_name=t['distruption_place_name']
        distruption_reason=t['distruption_reason']
        distruption_time=int(t['distruption_time'])
        actual_arrival_time=t['actual_arrival_time']

        distruption_time1 = int(distruption_time)
        if distruption_time1>=60:
            se='More Late'
        elif distruption_time1>=30 and distruption_time1<=60:
            se='Average Late'
        elif distruption_time1>=10 and distruption_time1<=30:
            se='Less Late'
        rail_delay_prediction_model.objects.create(names=names,rail_name=rail_name,rail_type=rail_type,departure_place=departure_place,
        destination=destination,
        departure_date=departure_date,
        departure_time=departure_time,
        arrival_date=arrival_date,
        arrival_time=arrival_time,
        distruption_place_name=distruption_place_name,
        distruption_reason=distruption_reason,
        distruption_time=distruption_time,
        actual_arrival_time=actual_arrival_time,
        impact=se
        )

    obj =rail_delay_prediction_model.objects.all()
    return render(request, 'SProvider/Train_View_Rail_DataSets_Details.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =impact_ratio_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})

def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="TrainedData.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = rail_delay_prediction_model.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1
        ws.write(row_num, 0, my_row.names, font_style)
        ws.write(row_num, 1, my_row.rail_name, font_style)
        ws.write(row_num, 2, my_row.rail_type, font_style)
        ws.write(row_num, 3, my_row.departure_place, font_style)
        ws.write(row_num, 4, my_row.destination, font_style)
        ws.write(row_num, 5, my_row.departure_time, font_style)
        ws.write(row_num, 6, my_row.arrival_date, font_style)
        ws.write(row_num, 7, my_row.arrival_time, font_style)
        ws.write(row_num, 8, my_row.distruption_place_name, font_style)
        ws.write(row_num, 9, my_row.distruption_reason, font_style)
        ws.write(row_num, 10, my_row.distruption_time, font_style)
        ws.write(row_num, 11, my_row.actual_arrival_time, font_style)
        ws.write(row_num, 12, my_row.actual_arrival_time, font_style)
        ws.write(row_num, 13, my_row.impact, font_style)
    wb.save(response)
    return response

















