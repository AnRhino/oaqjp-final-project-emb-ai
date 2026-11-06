"""
This module runs a Flask web application for Emotion Detection using IBM Watson NLP service.
"""
from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")


@app.route("/emotionDetector")
def sent_analyzer():
    """
    This function analyzes the text input for emotions 
    using the emotion_detector module and returns a formatted string.
    """
    # Lấy đoạn text từ URL query parameter
    text_to_analyze = request.args.get("textToAnalyze")

    response = emotion_detector(text_to_analyze)

    # Xử lý lỗi khi đầu vào không hợp lệ hoặc rỗng
    if not response or response.get("dominant_emotion") is None:
        return "Invalid text! Please try again!"

    return (
        f"For the given statement, the system response is 'anger': {response['anger']}, "
        f"'disgust': {response['disgust']}, 'fear': {response['fear']}, "
        f"'joy': {response['joy']} and 'sadness': {response['sadness']}. "
        f"The dominant emotion is {response['dominant_emotion']}."
    )


@app.route("/")
def render_index_page():
    """This function renders the main application index page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
