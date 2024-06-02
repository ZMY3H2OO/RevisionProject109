import numpy as np
from scipy import stats
from math import factorial
import matplotlib.pyplot as plt
import pandas as pd
import random
import csv

data = []
with open("CleanedData.csv", "r") as file:
    for line in file:
        l=line.strip().split(",")
        data.append(l)        
data.pop(0)

def filter_col(col_n, data, is_int):
    # Make a column of chart to a list
    new = []
    if is_int:
        for i in data:
            new.append(int(i[col_n]))
    else:
        for i in data:
            new.append(i[col_n])
    return new

# put grade, favorite projec, and previous experience into a list
grades = filter_col(2, data, True)
fav = filter_col(3, data, False)
prev = filter_col(4, data, False)

def get_exp(exp, exps):
    # check whether student answered a certain experience
    if exp in exps:
        return "1"
    else:
        return "0"
def filter_exp(exps):
    # make a string represent the previous experiences
    if exps == "None":
        return "00000"
    new = ""
    lst = ["Scratch", "Turtle", "Python", "Roblox", "HTML"]
    for i in lst:
        new += get_exp(i, exps)
    return new   
    
#convert all students' previous experiences to a string of binary numbers
prev_b = []
for i in prev:
    prev_b.append(filter_exp(i))

all, rev, scores = [], [], []
for i in data:
    new = i[5:13]
    for j in range(len(new)):
        new[j] = int(new[j]) 
    all.append(new)
    rev.append([new[0], new[2],new[4],new[6]])
    scores.append([new[1], new[3],new[5],new[7]])

"""
Question 1: Will grade influence the probability of revision?
"""
g7, g8 = [], []
for i in range(len(grades)):
    if grades[i] == 7:
        g7 += rev[i]
    else:
        g8 += rev[i]

def calc_diff(a, b):
    # return mean and observed difference of 2 groups
    meanA = np.mean(a)
    meanB = np.mean(b)
    diff_AB = abs(meanA-meanB)
    return meanA, meanB, diff_AB

print("Given this a student is in 7th grade, the probability of getting a satisfying grade without revision is", round(g7.count(0)/len(g7), 4))
print("Compared to 8th grade, it is", round(g8.count(0)/len(g8), 4))
mean7, mean8, diff_78 = calc_diff(g7, g8)
print("Mean revision times for 7th graders is", round(mean7, 4)) 
print("Mean revision times for 8th graders is", round(mean8, 4))
print("Observed Difference is", round(diff_78, 4))

def draw_sample(source, n):
    new = []
    for i in range(n):
        new.append(random.choice(source))
    return new
def p_value(a, b, diff):
    #bootstrap p value 
    count = 0
    uni = a+b
    for i in range(10000):
        a_sample = draw_sample(uni, len(a))
        b_sample = draw_sample(uni, len(b))
        meanA_B = abs(np.mean(a_sample)-np.mean(b_sample))
        if meanA_B >= diff:
            count += 1
    return count/10000

print("p-value for grade 7 to 8 is", p_value(g7, g8, diff_78))
print()

"""
Question 2: Will favorite project influence the probability of revision?
"""

def calc_fav(fav):
    #return a list of revision times for each student's favourite project and non-favorite
    new = []
    non_fav = []
    projects = ["1", "2", "3", "4"]
    for i in range(len(fav)):
        if "Final" in fav[i]:
            new.append(0)
            non_fav += rev[i]
        else:
            for j in range(4):
                if projects[j] in fav[i]:
                    new.append(rev[i][j])
                    non_fav += rev[i][:j] + rev[i][j+1:]
    return new, non_fav

fav_rev, nfav_rev = calc_fav(fav)
print("Given this a student's favorite project, the probability of getting a satisfying grade without revision is", round(fav_rev.count(0)/len(fav_rev), 4))
print("Compared to non-favorite projects, it is", round(nfav_rev.count(0)/len(nfav_rev), 4))

meanf, mean_nf, diff_f = calc_diff(fav_rev, nfav_rev)
print("Mean for favorite and non-favorite projects revision are", round(meanf, 4), round(mean_nf,4))
print("Observed Difference is", round(diff_f, 4))
print("p-value for favorite and non-favorite projects revision is", p_value(fav_rev, nfav_rev, diff_f))
print()

"""
Question 3: Will prior experience influence the probability of revision?
Specifically, given prior experience, will they revise less on things they knew? 
"""

def set_target(project):
#Project 1: scratch Project 2 & 3: turtle and python, project 4: roblox 
    target, end = 0, 0
    if project == 1:
        target = 0
        end = 1
    elif project == 2 or project == 3:
        target = 1
        end = 3
    else:
        target = 3
        end = 4
    return target, end

def calc_prev(prev, project):
    #return a list of revision times for each student's prior experience
    new = []
    non_prev = []
    target, end = set_target(project)
    
    for i in range(len(prev)):
        rev_time = rev[i][project-1]
        if "1" in prev[i][target:end]:
            new.append(rev_time)
        else:
            non_prev.append(rev_time)
    return new, non_prev

def p_experience(prev_b):
    for i in range(4):
        exp,n_exp = calc_prev(prev_b, i+1)
        mean_e, mean_ne, diff_exp = calc_diff(exp, n_exp)
        print("Observed Difference is", round(diff_exp, 4))
        print(f"p-value for experience and non-experience in project{i+1} revision is", p_value(exp, n_exp, diff_exp))
        print()
        
p_experience(prev_b)

"""
Apparently I am an effective teacher that prepare all students to the same level before doing their project based on all these high p-values:)

Question 4: For my own well-being in next semester, how many times I need to regrade students' homework for giving them infinite opportunity to revise? 

"""
#Assume all students' revision are individual and identical Poisson distribution with lambda = mean. 
sum_rev = []
for i in rev:
    sum_rev += i
mean_rev = np.mean(sum_rev)*4 

#Find how many revisions I will expect to get for a week given n students enrolled next semester. 
N = input("How many students enroll for next semester?")
N = "41" # I asked the registrar:)
expected_val = int(N)*mean_rev
#Find the probability that I will have a cozy semester under 100 revisions. 
p = stats.poisson.cdf(100, expected_val)
print(f"Expected revisions given {N} students next year under Poission is{expected_val}")
print("The probability of having a cozy semester under 100 revisions is", p)

#Calculate beta distribution after assuming a uniform prior of Beta(1, 1)
fail = np.sum(sum_rev)
print(f"Updated posterior belief of the probability of getting a satisfying grade is Beta(161, {fail+1})")
# updated expected value how many submissions needed to get 4 successes for each student
def B(a, b): 
    return factorial(a - 1) * factorial(b - 1) / factorial(a + b -1)
up_ep = B(160, fail+1)/B(161, fail+1)
exp = int(N) * 4  * (up_ep-1) # exclude successful submissions
print("Updated expected number of revisions for next semester is",round(exp, 2))
print()

"""
Question 5: Are student with more revisions correlate to better grades or worse?
"""

def sum_each(data):
# Sum each students' revisions and total grade
    new = []
    for i in data:
        new.append(np.sum(i))
    return new
    
rev_student = np.array(sum_each(rev))
scores_student = np.array(sum_each(scores))
#Printing the correlation coefficients:
x=np.corrcoef(rev_student, scores_student)
print(x)
#Do a linear regression and create a plot for visualization, modified from example from scipy library
result = stats.linregress(rev_student, scores_student)
plt.plot(rev_student, scores_student, 'o', label='original data')
plt.plot(rev_student, result.intercept + result.slope*rev_student, 'r', label='fitted line')
plt.legend()
plt.show()

"""
Just curious
Question 6: Given a student's prior experience, how accurate can I predict the favorite project with Naive Bayes?
Credit: Most codes come from Pset 6 question 4
"""
def write_csv(filename, rows):
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

fields = ["Scratch", "Turtle", "Python", "Roblox", "HTML", "Is Favorite"]
#write data to csv
for proj in ["1", "2", "3", "4", "Final"]:
    train = []
    test = []
    all_fav = []
    for i in range(len(prev_b)):     
        pre = list(prev_b[i])
        if proj in fav[i]:
            pre.append("1")
        else:
            pre.append("0")
        all_fav.append(pre)
    #randomly pick 8 people to be in the test group
    test = []
    for i in range(8):
        choice = random.choice(all_fav)
        all_fav.remove(choice)
        test.append(choice)

    write_csv(f"Project{proj}-train.csv", all_fav)
    write_csv(f"Project{proj}-test.csv", test)
    
#Following are almost the same as my answer for Pset6 question 4 to operate Naive Bayes
class NaiveBayes:

    def __init__(self, dataset_name):
        '''
        Configures the classifier so that it's able to train itself using the
        named dataset and ultimately make predictions.
        '''
        self.dataset_name = dataset_name
        self.label_counts = [0, 0]
        self.feature_counts = {}
        
    def train(self):
        # Train classifier by compiling a collection of label
        # counts and feature counts.
        grid = pd.read_csv(self.dataset_name + "-train.csv").to_numpy()
        train_features = grid[:,:-1]
        train_labels = grid[:, -1].T

        count0, count1 = 0, 0
        d = {}
        l = len(train_features[0])

        for i in range(l):
            d[f"feature{i}"] = [[0, 0],[0, 0]]
        for i in range(len(train_features)):
            for j in range(l):
                if train_labels[i] == 0 and train_features[i][j] == 0:
                    d[f"feature{j}"][0][0] += 1
                    count0 += 1
                elif train_labels[i] == 0 and train_features[i][j] == 1:
                    d[f"feature{j}"][0][1] += 1
                    count0 += 1
                elif train_labels[i] == 1 and train_features[i][j] == 0:
                    d[f"feature{j}"][1][0] += 1
                    count1 += 1
                else:
                    d[f"feature{j}"][1][1] += 1
                    count1 += 1
        self.label_counts = [count0, count1]
        self.feature_counts = d

    def predict_one(self, features):
        prediction = 0
        p0, p1 = self.py0, self.py1
        n_feature = len(self.feature_counts)
        count_y0 = self.label_counts[0] + 2*n_feature
        count_y1 = self.label_counts[1] + 2*n_feature
       
        for i in range(len(features)):
            if features[i] == 0:
                p0 *= (self.feature_counts[f"feature{i}"][0][0]+1)/(count_y0)
                p1 *= (self.feature_counts[f"feature{i}"][1][0]+1)/(count_y1)
            else:
                p0 *= (self.feature_counts[f"feature{i}"][0][1]+1)/(count_y0)
                p1 *= (self.feature_counts[f"feature{i}"][1][1]+1)/(count_y1)

        if p1 > p0:
            prediction = 1
        
        return prediction

    def predict_all(self, rows):
        preds = np.zeros(rows.shape[0], dtype=np.uint8)
        # calculate p(Y =0) and p(Y =1) here
        total_y= sum(self.label_counts)
        self.py0 = self.label_counts[0]/total_y
        self.py1 = self.label_counts[1]/total_y
        self.x1 = []
        for row_index, row in enumerate(rows):
            preds[row_index] = self.predict_one(row)
        return preds
    
    def accuracy(self, suffix):
        grid = pd.read_csv(self.dataset_name + "-" + suffix + ".csv").to_numpy()
        rows = grid[:, :-1]
        actual_labels = grid[:,-1]
        predicted_labels = self.predict_all(rows)
        matches = 0
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == actual_labels[i]:
                matches += 1
        return matches / len(predicted_labels)

for dataset_name in ["Project1", "Project2", "Project3", "Project4", "Projectfinal"]:
    classifier = NaiveBayes(dataset_name)
    classifier.train()
    print("Dataset name: {}, train accuracy: {:.3f}".format(dataset_name, classifier.accuracy("train")))
    print("Dataset name: {}, test accuracy: {:.3f}".format(dataset_name, classifier.accuracy("test")))
    print("For someone without any experience in all languages in this class, the prediction of choosing this project as favorite is", classifier.predict_one([0,0,0,0,0]))
