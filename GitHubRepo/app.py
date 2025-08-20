import os
import uuid
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename

# Import our custom detectors
from text_processor import extract_text_from_file, clean_up_text, get_text_info
from plagiarism_detector import PlagiarismDetector
from ai_detector import AIDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my-super-secret-password-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///integrity.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detectors
print("🔄 Initializing detectors with custom sensitivity...")
plagiarism_detective = PlagiarismDetector(
    ref_folder="reference_texts",
    semantic_threshold=0.3  # More sensitive threshold for testing
)
ai_detective = AIDetector(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_tokens=512,
    stride=50,
    ai_weight=0.6,
    pattern_weight=0.4,
    high_threshold=85,
    medium_threshold=65
)
print("✅ All detectives are ready!")

db = SQLAlchemy(app)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    assignment_name = db.Column(db.String(255), nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    plagiarism_score = db.Column(db.Float, default=0.0)
    ai_score = db.Column(db.Float, default=0.0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    analysis_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))

class UploadForm(FlaskForm):
    file = FileField('Choose Document', validators=[\
        FileAllowed(['txt', 'pdf', 'docx'], 'Only TXT, PDF, and DOCX files are allowed!')\
    ])
    assignment_name = StringField('Assignment Name', validators=[DataRequired()], default='My Assignment')
    submit = SubmitField('Upload & Analyze')

@app.route('/', methods=['GET', 'POST'])
def home_page():
    form = UploadForm()
    if form.validate_on_submit():
        files = request.files.getlist('file')
        assignment_name = form.assignment_name.data.strip()

        # Bulk-upload flow
        if len(files) > 1:
            uploaded, rejected = [], []
            for f in files[:150]:
                raw_name = f.filename or ''
                filename = secure_filename(raw_name)
                reason = None

                if not filename:
                    reason = 'invalid filename'
                else:
                    name, ext = os.path.splitext(filename.lower())
                    if ext not in {'.pdf', '.docx', '.txt'}:
                        reason = 'unsupported file type'

                if reason:
                    rejected.append(f"{raw_name} ({reason})")
                    continue

                unique = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
                path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique)
                f.save(path)

                text = extract_text_from_file(path)
                if len(text.strip()) < 100:
                    rejected.append(f"{filename} (content too short)")
                    os.remove(path)
                    continue

                processed = clean_up_text(text)
                plag_score = plagiarism_detective.check_for_plagiarism(processed)['overall_score']
                ai_score = ai_detective.detect_ai_content(processed)['ai_probability']

                record = Analysis(
                    filename=filename,
                    assignment_name=assignment_name,
                    original_text=text,
                    plagiarism_score=plag_score,
                    ai_score=ai_score
                )
                db.session.add(record)
                uploaded.append(filename)
                os.remove(path)

            db.session.commit()
            flash(f'Bulk upload complete: {len(uploaded)} succeeded, {len(rejected)} rejected.', 'success')
            if rejected:
                flash('Rejected: ' + '; '.join(rejected), 'error')
            return redirect(url_for('show_history'))

        # Single-file flow
        try:
            f = files[0]
            filename = secure_filename(f.filename)
            name, ext = os.path.splitext(filename)
            unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            f.save(filepath)

            original_text = extract_text_from_file(filepath)
            if len(original_text.strip()) < 100:
                flash('Document too short—please upload at least 100 characters.', 'error')
                os.remove(filepath)
                return redirect(url_for('home_page'))

            processed_text = clean_up_text(original_text)
            print("🕵️ Starting plagiarism analysis...")
            plagiarism_result = plagiarism_detective.check_for_plagiarism(processed_text)
            print("🤖 Starting AI detection...")
            ai_result = ai_detective.detect_ai_content(processed_text)

            analysis = Analysis(
                filename=filename,
                assignment_name=assignment_name,
                original_text=original_text,
                plagiarism_score=plagiarism_result['overall_score'],
                ai_score=ai_result['ai_probability']
            )
            db.session.add(analysis)
            db.session.commit()
            os.remove(filepath)

            flash('Analysis completed successfully!', 'success')
            return redirect(url_for('show_results', analysis_id=analysis.analysis_id))

        except Exception as e:
            flash(f'Oops! Something went wrong: {e}', 'error')
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('home_page'))

    recent_analyses = Analysis.query.order_by(Analysis.timestamp.desc()).limit(5).all()
    system_info = {
        'ai_detective_working': ai_detective.model_working,
        'total_analyses_done': Analysis.query.count()
    }
    return render_template('index.html', form=form, recent_analyses=recent_analyses, system_info=system_info)

@app.route('/results/<analysis_id>')
def show_results(analysis_id):
    analysis = Analysis.query.filter_by(analysis_id=analysis_id).first_or_404()
    processed = clean_up_text(analysis.original_text)
    plagiarism_result = plagiarism_detective.check_for_plagiarism(processed)
    ai_result = ai_detective.detect_ai_content(processed)
    text_stats = get_text_info(analysis.original_text)
    return render_template('results.html',
                           analysis=analysis,
                           plagiarism_result=plagiarism_result,
                           ai_result=ai_result,
                           text_stats=text_stats)

@app.route('/history')
def show_history():
    page = request.args.get('page', 1, type=int)
    analyses = Analysis.query.order_by(Analysis.timestamp.desc()) \
        .paginate(page=page, per_page=20, error_out=False)
    return render_template('history.html', analyses=analyses)

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.errorhandler(413)
def file_too_large(e):
    flash("File is too large! Maximum size is 32MB.", "error")
    return redirect(url_for('home_page'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)