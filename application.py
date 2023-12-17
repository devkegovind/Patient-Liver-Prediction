from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')



@app.route('/predict', methods = ['GET', 'POST'])

def predict_datapoint():
    if request.method == "GET":
        return render_template('form.html')
    
    else:
        data = CustomData(
            Age = int(request.form.get('Age')),
            Gender = request.form.get('Gender'),
            Total_Bilirubin = float(request.form.get('Total_Bilirubin')),
            Direct_Bilirubin = float(request.form.get('Direct_Bilirubin')),
            Alkaline_Phosphotase = int(request.form.get('Alkaline_Phosphotase')),
            Alamine_Aminotransferase = int(request.form.get('Alamine_Aminotransferase')),
            Aspartate_Aminotransferase = int(request.form.get('Aspartate_Aminotransferase')),  
            Total_Protiens = float(request.form.get('Total_Protiens')),
            Albumin = float(request.form.get('Albumin')),
            Albumin_and_Globulin_Ratio = float(request.form.get('Albumin_and_Globulin_Ratio'))
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = int((pred[0]))

        return render_template('results.html', final_result = results)
    

if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug = True)
