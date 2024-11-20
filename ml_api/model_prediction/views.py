from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import joblib
from .serializers import PredictionInputSerializer

# Load the trained model
model = joblib.load('model/sales_model_2024-09-23-16-22-02.pkl')

class ModelPredictionView(APIView):
    """
    API endpoint for making sales predictions.
    """
    def post(self, request, *args, **kwargs):
        serializer = PredictionInputSerializer(data=request.data)
        if serializer.is_valid():
            try:
                # Prepare input data
                input_df = pd.DataFrame([serializer.validated_data])

                # Preprocess the input data (if needed)
                # Example: processed_data = preprocess_input(input_df)
                processed_data = input_df  # Replace with preprocessing function if needed

                # Predict
                prediction = model.predict(processed_data)
                prediction_value = round(prediction[0], 2)

                # Return response
                return Response(
                    {"store_id": serializer.validated_data['Store'], "predicted_sales": prediction_value},
                    status=status.HTTP_200_OK
                )
            except Exception as e:
                return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
