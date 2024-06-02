import numpy as np
from scipy import stats
from math import factorial
import matplotlib.pyplot as plt
import random

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
# print("Mean revision times for 7th graders is", round(mean7, 4)) 
# print("Mean revision times for 8th graders is", round(mean8, 4))
print("Observed Difference is", round(diff_78, 4))
print()

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

#print("p-value for grade 7 to 8 is", p_value(g7, g8, diff_78))

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

# meanf, mean_nf, diff_f = calc_diff(fav_rev, nfav_rev)
# print("Mean for favorite and non-favorite projects revision are", round(meanf, 4), round(mean_nf,4))
# print("Observed Difference is", round(diff_f, 4))
#print("p-value for favorite and non-favorite projects revision is", p_value(fav_rev, nfav_rev, diff_f))
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
        #print("Observed Difference is", round(diff_exp, 4))
        #print(f"p-value for experience and non-experience in project{i+1} revision is", p_value(exp, n_exp, diff_exp))
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
print(mean_rev)

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
# up_ep = B(160, fail+1)/B(161, fail+1)
# exp = int(N) * 4  * (up_ep-1) # exclude successful submissions
# print("Updated expected number of revisions for next semester is",round(exp, 2))

#Are student with more revisions correlate to better grades?
def sum_each(data):
# Sum each students' revisions and total grade
    new = []
    for i in data:
        new.append(np.sum(i))
    return new
    
# rev_student = np.array(sum_each(rev))
# scores_student = np.array(sum_each(scores))
# #Printing the correlation coefficients:
# x=np.corrcoef(rev_student, scores_student)
# print(x)
# #Do a linear regression and create a plot for visualization, modified from example from scipy library
# result = stats.linregress(rev_student, scores_student)
# plt.plot(rev_student, scores_student, 'o', label='original data')
# plt.plot(rev_student, result.intercept + result.slope*rev_student, 'r', label='fitted line')
# plt.legend()
# plt.show()
