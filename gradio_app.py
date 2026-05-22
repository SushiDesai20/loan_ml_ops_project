import gradio as gr
import joblib
import numpy as np

model=joblib.load("loan_approval_pipeline.pkl")

def predict_loan(
        no_of_dependents, 
        education, 
        self_employed, 
        income_annum, 
        loan_amount, 
        loan_term, 
        cibil_score, 
        residential_assets_value, 
        commercial_assets_value, 
        luxury_assets_value, 
        bank_asset_value
    ):


    data=np.array([[no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]])
   
    prediction_loan=model.predict(data)
    probability=model.predict_proba(data)
    confidence=round(np.max(probability)*100,2)

    if prediction_loan[0]==1:
        return f'Loan Approved with confidence {confidence} %'
    else:
        return f'Loan Rejected with confidence {confidence} %'

iface=gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Number(label="Number of Dependents"),
        gr.Number(label='Education(0-Educated | 1-Not Graduated)'),
        gr.Number(label='Self Employment(0=No | 1=Yes)'),
        gr.Number(label="Income Annually"),
        gr.Number(label="Loan Amount"),
        gr.Number(label="Loan Term"),
        gr.Number(label="Cibil Score"),
        gr.Number(label="Residential Assets Value"),
        gr.Number(label="Commercial Assets Value"),
        gr.Number(label="Luxury Assets Value"),
        gr.Number(label="Bank Asset Value")
    ],
    outputs=gr.Textbox(label='Loan Approval Prediction'),
    title='Loan Approval Prediction'
)
    
iface.launch(debug=True)