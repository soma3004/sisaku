from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(80), nullable=False)
    score = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<Score {self.id}: {self.subject} - {self.score}>'

from flask import Flask, render_template, request, redirect, url_for
from models import db, Score

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scores.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/')
def index():
    scores = Score.query.all()
    total_score = sum(score.score for score in scores)
    return render_template('scores.html', scores=scores, total_score=total_score)

@app.route('/add', methods=['POST'])
def add_score():
    subject = request.form['subject']
    score = request.form['score']
    new_score = Score(subject=subject, score=int(score))
    db.session.add(new_score)
    db.session.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>テストの点数管理</title>
</head>
<body>
    <h1>テストの点数</h1>
    <div id="scores">
        <table>
            <tr>
                <th>教科</th>
                <th>点数</th>
            </tr>
            {% for score in scores %}
            <tr>
                <td>{{ score.subject }}</td>
                <td>{{ score.score }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <h2>総合点: {{ total_score }}</h2>

    <form action="/add" method="POST">
        <input type="text" name="subject" placeholder="教科名" required>
        <input type="number" name="score" placeholder="点数" required>
        <button type="submit">追加</button>
    </form>
</body>
</html>

