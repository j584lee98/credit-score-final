import pickle
import pandas as pd
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras.models import load_model

month = int(input('Enter the month (int): '))
age = int(input('Enter the age (int): '))
annual_income = float(input('Enter the annual income (float): '))
monthly_inhand_salary = float(input('Enter the monthly in-hand salary (float): '))
num_bank_accounts = int(input('Enter the number of bank accounts (int): '))
num_credit_card = int(input('Enter the number of credit cards (int): '))
interest_rate = float(input('Enter the interest rate (float): '))
num_of_loan = int(input('Enter the number of loans (int): '))
delay_from_due_date = int(input('Enter the delay from payment due date in days (int): '))
num_of_delayed_payment = int(input('Enter the number of delayed payments (int): '))
changed_credit_limit = int(input('Enter the number of credit limit changes (int): '))
num_credit_inquiries = int(input('Enter the number of credit inquiries (int): '))
credit_mix = int(input('Enter a value (0-2, ascending quality) for credit mix quality (str): '))
outstanding_debt = float(input('Enter the outstanding debt (float): '))
credit_utilization_ratio = float(input('Enter the credit utilization ratio (float): '))
payment_of_min_amount = int(input('Enter 1 if minimum amount is paid, 0 if not (int): '))
total_EMI_per_month = float(input('Enter the total EMI per month (float): '))
amount_invested_monthly = float(input('Enter the amount invested monthly (float): '))
payment_behaviour = int(input('Enter a value (0-5, ascending amount) for spending behaviour and payments (int): '))
monthly_balance = float(input('Enter the monthly balance (float): '))
occupation = input('Enter the occupation (str): ')

infer_data = pd.DataFrame({
    'Month': [month],
    'Age': [age],
    'Annual_Income': [annual_income],
    'Monthly_Inhand_Salary': [monthly_inhand_salary],
    'Num_Bank_Accounts': [num_bank_accounts],
    'Num_Credit_Card': [num_credit_card],
    'Interest_Rate': [interest_rate],
    'Num_of_Loan': [num_of_loan],
    'Delay_from_due_date': [delay_from_due_date],
    'Num_of_Delayed_Payment': [num_of_delayed_payment],
    'Changed_Credit_Limit': [changed_credit_limit],
    'Num_Credit_Inquiries': [num_credit_inquiries],
    'Credit_Mix': [credit_mix],
    'Outstanding_Debt': [outstanding_debt],
    'Credit_Utilization_Ratio': [credit_utilization_ratio],
    'Payment_of_Min_Amount': [payment_of_min_amount],
    'Total_EMI_per_month': [total_EMI_per_month],
    'Amount_invested_monthly': [amount_invested_monthly],
    'Payment_Behaviour': [payment_behaviour],
    'Monthly_Balance': [monthly_balance],
    'Occupation_Accountant': [1 if occupation == 'Accountant' else 0],
    'Occupation_Architect': [1 if occupation == 'Architect' else 0],
    'Occupation_Developer': [1 if occupation == 'Developer' else 0],
    'Occupation_Doctor': [1 if occupation == 'Doctor' else 0],
    'Occupation_Engineer': [1 if occupation == 'Engineer' else 0],
    'Occupation_Entrepreneur': [1 if occupation == 'Entrepreneur' else 0],
    'Occupation_Journalist': [1 if occupation == 'Journalist' else 0],
    'Occupation_Lawyer': [1 if occupation == 'Lawyer' else 0],
    'Occupation_Manager': [1 if occupation == 'Manager' else 0],
    'Occupation_Mechanic': [1 if occupation == 'Mechanic' else 0],
    'Occupation_Media_Manager': [1 if occupation == 'Media_Manager' else 0],
    'Occupation_Musician': [1 if occupation == 'Musician' else 0],
    'Occupation_Scientist': [1 if occupation == 'Scientist' else 0],
    'Occupation_Teacher': [1 if occupation == 'Teacher' else 0],
    'Occupation_Writer': [1 if occupation == 'Writer' else 0],
})

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

scaled_data = scaler.transform(infer_data)

model = load_model('model.keras')

y_pred = model.predict(scaled_data)

credit_score_dict = {
    'Poor': 0,
    'Standard': 1,
    'Good': 2
}

print('Probabilities ->')
print(f'Good: {y_pred[0][credit_score_dict['Good']]:.1%}')
print(f'Standard: {y_pred[0][credit_score_dict['Standard']]:.1%}')
print(f'Poor: {y_pred[0][credit_score_dict['Poor']]:.1%}')
print('\nPrediction ->')

y_pred = np.argmax(y_pred, axis=1)
score = [k for k,v in credit_score_dict.items() if v == y_pred][0]

print(score)