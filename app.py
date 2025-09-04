from src.pipelines.prediction_pipeline import PredictPipeline, CustomData

obj = CustomData(
  2.83,0.41,2.86,0.05,0.51,16.86
)
frame = obj.get_data_as_dataframe()

pred_obj = PredictPipeline()

result = pred_obj.predict_cl(frame)
print(f"P-wave detection: {result==1}")


if result[0] == 1:
     ttf_result = pred_obj.predict_rg(frame)
     print(f"Time-to-failure: {ttf_result}")
