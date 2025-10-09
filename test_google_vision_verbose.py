from inference_sdk import InferenceHTTPClient
import streamlit as st

api_key = st.secrets.get("ROBOFLOW_API_KEY", "")
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

result = CLIENT.infer("610pu5al5hL.jpg", model_id="garbage-classification-3/2")
print(result)